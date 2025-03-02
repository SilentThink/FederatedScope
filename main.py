import argparse
import os
import time
import random
import numpy as np
import torch
from server import Server
from client import Client
from utils_data.load_data import get_loaders, record_client_datasets

import yaml
from copy import deepcopy
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入投毒攻击模块
from poisoning_attack import PerturbationPoisoningAttack
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Federation
    parser.add_argument('--num_clients', type=int, default=200, help='N in our paper')
    parser.add_argument('-m', type=float, default=0.05, help='ratio of activate clients in each round')
    parser.add_argument('--rounds', type=int, default=40, help='the total number of rounds')
    parser.add_argument('--local_step', type=int, default=200, help=r'$\tau in our paper')
    parser.add_argument('--batch_or_epoch', type=str, default='batch', choices=['epoch', 'batch'])
    parser.add_argument('--equal_weight', default=False, action='store_true', help='if `true`, the weights among clients for aggregation are the same')

    # Data
    ## Arguments related to data on both datasets
    parser.add_argument('--dataset', type=str, default='instruct', choices=['instruct', 'dolly'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size > 1 may cause error during running')
    parser.add_argument('--max_length', type=int, default=512, help='the max number of tokens of a data instance')
    parser.add_argument('--use_prompts', default=True, help='if `true`, the prompt template from alpaca is adopted')
    
    ## Arguments related to data only for Dolly-15K
    parser.add_argument('--iid', type=str, default='dir0.5', help=r'`dir{alpha}` means that \alpha in Dirichlet distribution, `0` means IID split')
    parser.add_argument('--zerotask', default=7, type=int, help='the index of the task for evaluation in dolly-15K')
    parser.add_argument('--dataset_subsample', type=float, default=1.0, help='used for sampling a subset from the original dataset, only effective for dolly-15K')

    # Model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help=r'learning rate \eta')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay in MeZO')
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')

    # Training args only for `FedKSeed`
    parser.add_argument('-K', type=int, default=4096, help='ratio of active clients in each round')
    parser.add_argument('--zo_eps', type=float, default=0.0005, help=r'\eps in MeZO')

    # Training args only for `FedKSeed-Pro`
    parser.add_argument('--bias_sampling', default=False, action='store_true', help='if `true`, the probabilities of candidate seeds to be sampled are not identical, i.e., FedKSeed-Pro')
    parser.add_argument('--bias_loss_clip', default=1000.0, type=float, help='scalar gradient whose abstract values exceeds this value will be cliped')
    parser.add_argument('--grad_initial', default=0.0, type=float, help='initial value of scalar gradient history corresponding to each candidate seed')

    # Environment
    parser.add_argument('--device', type=int, default=0, help='index of the targeted cuda device')
    parser.add_argument('--log', default=False, action='store_true', help='if `true`, running logs will be recorded in files')
    parser.add_argument('--log_root', default='logs', help='root path of log files')
    parser.add_argument('--seed', default=42, type=int, help='global seed, for reproducibility')
    
    # Evaluation
    parser.add_argument('--eval_metric', default='rouge', type=str, choices=['rouge', 'loss'], help='metric to evaluate global model in the last round')
    
    # Checkpoints
    parser.add_argument('--save', default=False, action='store_true', help='if `true`, the checkpoint of tuned models will be stored')

    # 解析命令行参数
    parser.add_argument('--attack', action='store_true', help='是否进行投毒攻击')
    parser.add_argument('--target_client', type=int, default=0, help='目标客户端索引')
    parser.add_argument('--k1', type=int, default=20, help='相似度筛选后的候选数量')
    parser.add_argument('--k2', type=int, default=10, help='最终选择的投毒序列数量')

    # 添加新的参数来控制生成
    parser.add_argument('--gen_max_length', type=int, default=256, help='maximum length for generation')
    parser.add_argument('--gen_min_length', type=int, default=1, help='minimum length for generation')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams for beam search')

    # 在主函数中添加投毒间隔参数
    parser.add_argument('--poison_interval', type=int, default=5, help='投毒间隔，每隔多少轮进行一次投毒')

    # 在参数解析部分添加 lr_decay 参数
    parser.add_argument('--lr_decay', type=float, default=1.0, help='学习率衰减系数')

    time_stamp = str(time.time())
    args = parser.parse_args()

    eval_avg_acc = []
    memory_record_dic = {}
    
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    setup_seed(args.seed)
    list_train_loader, eval_loader, _ = get_loaders(args)
    
    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')
    if args.log:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as writer:
            writer.write(config)

    # 修改客户端选择逻辑，确保目标客户端每轮都参与训练
    def select_clients_for_round(num_clients, m, target_client_idx=None):
        """选择参与本轮训练的客户端"""
        num_selected = max(1, int(num_clients * m))
        
        if target_client_idx is not None:
            # 确保目标客户端被选中
            remaining_slots = num_selected - 1
            other_clients = [i for i in range(num_clients) if i != target_client_idx]
            
            if remaining_slots > 0 and len(other_clients) > 0:
                selected_others = np.random.choice(other_clients, 
                                                  size=min(remaining_slots, len(other_clients)), 
                                                  replace=False)
                return np.append(selected_others, target_client_idx)
            else:
                return np.array([target_client_idx])
        else:
            # 正常随机选择
            return np.random.choice(np.arange(num_clients), size=min(num_selected, num_clients), replace=False)

    # 在主函数中替换客户端选择逻辑
    client_indices_rounds = []
    for _ in range(args.rounds):
        if args.attack:
            # 确保目标客户端每轮都参与训练
            client_indices_rounds.append(select_clients_for_round(args.num_clients, args.m, args.target_client))
        else:
            client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.m), replace=False))

    client_list = []
    
    # sample `K` candidate seeds
    candidate_seeds = np.random.randint(1, 100000000000, args.K)

    server = Server(args, eval_loader=eval_loader, candidate_seeds=candidate_seeds, log_dir=log_dir)
    for idx in range(args.num_clients):
        client_list.append(Client(idx, args, candidate_seeds, list_train_loader[idx]))
    
    eval_result = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    eval_avg_acc.append(eval_result)
    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({
                'eval_avg_acc': eval_avg_acc
            }, writer)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 目标样本
    target_sample_str = """{"instruction": "Who are N-Dubz?", "context": "", "response": "N-Dubz are a popular band in the United Kingdom, made up of Tulisa, Fazer, and Dappy, formed out of London. The band were formed when they were young teenagers in the early 2000s. They were inspired to form the band by Dappy's late father, known to the band as \\"Uncle B\\". Their song \\"Papa can you hear me?\\" is a tribute to Uncle B. Tulisa and Dappy are cousins, whilst Fazer has always been a close friend.\\nThe trio have had many successful hits, and collaborated with popular artists like Tinchy Strider and Skepta. They parted ways in 2011, and Dappy started a solo career, whilst Tulisa became a judge on the popular UK show \\"The X Factor\\". She formed and mentored the winning band Little Mix. \\nThe band reunited in 2022 and released new music, along with a sold out UK tour.", "category": "general_qa"}"""

    # 初始化投毒攻击模块（如果需要）
    poisoning_attack = None
    if args.attack:
        poisoning_attack = PerturbationPoisoningAttack(
            args=args,
            server=server,
            target_sample_str=target_sample_str,
            k1=args.k1,
            k2=args.k2,
            poison_interval=args.poison_interval
        )

    # 在进行eval之前添加错误处理
    def safe_eval(server, cur_round, eval_avg_acc):
        try:
            with torch.amp.autocast(device_type='cuda'):
                return server.eval(cur_round=cur_round, eval_avg_acc=eval_avg_acc)
        except RuntimeError as e:
            print(f"警告: 评估时发生错误: {str(e)}")
            # 返回一个默认值或上一次的评估结果
            return eval_avg_acc[-1] if eval_avg_acc else 0.0

    for r in range(1, args.rounds + 1):
        selected_client = [client_list[i] for i in client_indices_rounds[r-1]]
        
        # 检查目标客户端是否在本轮被选中
        target_client = None
        if args.attack:
            for client in selected_client:
                if client.idx == args.target_client:
                    target_client = client
                    break
        
        if args.bias_sampling:
            probabilities = server.calculate_probabilities()
        else:
            probabilities = None
        
        # 根据投毒间隔决定是否进行投毒
        should_poison = args.attack and (r % args.poison_interval == 0)
        
        if should_poison:
            # 如果目标客户端被选中且需要进行攻击，对其进行投毒攻击
            print(f"对客户端 {args.target_client} 进行投毒攻击...")
            
            # 获取当前轮次使用的种子
            current_round_seeds = server.candidate_seeds
            
            # 对服务器种子池进行投毒
            poisoned_seed_pool = poisoning_attack.poison_server_seed_pool(
                args.target_client, current_round_seeds, tokenizer)
            
            # 使用投毒后的种子池创建模型
            poisoned_model = server.create_poisoned_model_by_seedpool(
                r, poisoned_seed_pool, target_client)
            
            # 目标客户端使用投毒模型训练
            response = target_client.local_train_with_seed_pool(
                poisoned_model, cur_round=r, 
                memory_record_dic=memory_record_dic, 
                probabilities=probabilities, 
                gradient_history=server.gradient_history)
            
            # 记录客户端响应
            poisoning_attack.record_client_response(args.target_client, response)
            
            # 其他客户端正常训练
            for client in selected_client:
                if client.idx != args.target_client:
                    client.local_train_with_seed_pool(
                        deepcopy(server.model), cur_round=r, 
                        memory_record_dic=memory_record_dic, 
                        probabilities=probabilities, 
                        gradient_history=server.gradient_history)
        else:
            # 所有客户端正常训练
            for client in selected_client:
                response = client.local_train_with_seed_pool(
                    deepcopy(server.model), cur_round=r, 
                    memory_record_dic=memory_record_dic, 
                    probabilities=probabilities, 
                    gradient_history=server.gradient_history)
                
                # 如果是目标客户端且进行攻击，记录响应
                if args.attack and client.idx == args.target_client:
                    poisoning_attack.record_client_response(args.target_client, response)
        
        # 聚合更新
        server.aggregate_seed_pool(selected_client)
        server.update_global_model_by_seed_pool()
        
        # 如果进行攻击，记录目标样本在当前模型上的损失
        if args.attack:
            loss = poisoning_attack.record_loss(args.target_client, r, server.model, tokenizer)
            print(f"轮次 {r}: 目标样本损失 = {loss:.4f}")
        
        # 使用安全的eval函数
        eval_result = safe_eval(server, cur_round=r, eval_avg_acc=eval_avg_acc)
        eval_avg_acc.append(eval_result)
        
        # 如果进行攻击且到达一定轮次，进行成员推断
        if args.attack and r % 5 == 0:
            membership_result = poisoning_attack.membership_inference(args.target_client)
            print(f"轮次 {r}: 目标样本成员身份推断结果: {membership_result}")

        # 在每轮结束时添加
        if args.attack:
            print(f"轮次 {r} 结束，目标客户端 {args.target_client} 的响应记录数: {len(poisoning_attack.client_responses.get(args.target_client, []))}")

    # 训练结束后，如果进行了攻击，绘制损失曲线
    if args.attack:
        save_path = os.path.join(log_dir, f'client_{args.target_client}_loss_curve.png')
        poisoning_attack.plot_loss_curve(args.target_client, save_path)
        
        # 将结果保存到文件
        if args.log:
            with open(os.path.join(log_dir, 'attack_results.json'), 'w') as f:
                json.dump({
                    'target_client': args.target_client,
                    'membership_result': poisoning_attack.membership_inference(args.target_client),
                    'loss_history': poisoning_attack.loss_history.get(args.target_client, {})
                }, f, indent=2)

    # 最终评估
    setup_seed(args.seed)
    _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    server.eval_loader = eval_loader_final
    
    # 使用安全的eval函数进行最终评估
    final_eval_result = safe_eval(server, cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    
    if args.log:
        with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
            json.dump({
                f'final_eval_{args.eval_metric}': final_eval_result
            }, writer)
    print(f'final round {args.eval_metric}: {final_eval_result}')

    # 在主函数中添加记录客户端数据集的代码
    if args.attack:
        # 记录客户端数据集中是否包含目标样本
        client_datasets = [client.train_loader.dataset for client in client_list]
        client_has_target = record_client_datasets(
            client_datasets, 
            target_sample_str, 
            os.path.join(log_dir, 'client_datasets.json')
        )
        print(f"客户端数据集记录已保存至 {os.path.join(log_dir, 'client_datasets.json')}")
        
        # 打印目标客户端是否包含目标样本
        if str(args.target_client) in client_has_target:
            print(f"目标客户端 {args.target_client} {'包含' if client_has_target[str(args.target_client)] else '不包含'} 目标样本")
