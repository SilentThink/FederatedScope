import torch
import numpy as np
from copy import deepcopy
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import matplotlib

class PerturbationPoisoningAttack:
    def __init__(self, args, server, target_sample_str, k1=20, k2=10, poison_interval=5):
        """
        初始化投毒攻击模块
        
        参数:
            args: 配置参数
            server: 服务器对象
            target_sample_str: 目标样本字符串
            k1: 相似度筛选后的候选数量
            k2: 最终选择的投毒序列数量
            poison_interval: 投毒间隔，每隔多少轮进行一次投毒
        """
        self.args = args
        self.server = server
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = 0.7  # 相似度判定阈值τ
        self.k1 = k1  # 相似度筛选后的候选数量
        self.k2 = k2  # 最终选择的投毒序列数量
        self.poison_interval = poison_interval  # 投毒间隔
        
        # 解析目标样本
        self.target_sample = json.loads(target_sample_str)
        
        # 记录客户端响应和损失历史
        self.client_responses = {}
        self.loss_history = {}
        
        # 保存目标样本到文件
        self.save_target_sample()
        
    def save_target_sample(self):
        """将目标样本保存到文件"""
        os.makedirs('attack_data', exist_ok=True)
        with open('attack_data/target_sample.json', 'w') as f:
            json.dump(self.target_sample, f, indent=2)
        print(f"目标样本已保存到 attack_data/target_sample.json")
    
    def prepare_target_sample_tensor(self, tokenizer):
        """将目标样本转换为模型输入格式"""
        if self.target_sample is None:
            return None
        
        # 构建输入文本
        instruction = self.target_sample.get("instruction", "")
        context = self.target_sample.get("context", "")
        response = self.target_sample.get("response", "")
        
        if context:
            input_text = f"Instruction: {instruction}\nContext: {context}\nResponse: "
        else:
            input_text = f"Instruction: {instruction}\nResponse: "
        
        full_text = input_text + response
        
        # 分词处理
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.args.max_length)
        
        # 创建标签，将非响应部分的标签设为-100（忽略）
        labels = inputs.input_ids.clone()
        
        # 计算响应开始的位置
        response_tokens = tokenizer(input_text, return_tensors="pt").input_ids[0]
        response_start = len(response_tokens)
        
        # 确保response_start不超过输入长度
        response_start = min(response_start, inputs.input_ids.size(1))
        
        # 将非响应部分的标签设为-100
        if response_start > 0:
            labels[0, :response_start] = -100
        
        # 添加到输入字典
        inputs['labels'] = labels
        
        return inputs
    
    # todo: 候选数量不足k1时需要根据与当前轮次种子的平均相似度选取
    def similarity_measurement(self, history_perturbations, current_round_seeds):
        """
        计算相似度并筛选候选投毒序列
        
        返回:
            candidates: 候选投毒序列字典 {seed: value}
        """
        candidates = {}  # 改为返回字典而不是列表
        
        # 计算相似度
        for s_i, v_i in history_perturbations.items():
            if s_i in current_round_seeds:
                # 添加到候选集
                candidates[s_i] = v_i
        
        # 如果候选数量超过k1，进行筛选
        if len(candidates) > self.k1:
            # 按照值的绝对大小排序
            sorted_candidates = sorted(candidates.items(), key=lambda x: abs(x[1]), reverse=True)
            # 只保留前k1个
            candidates = dict(sorted_candidates[:self.k1])
        
        return candidates
    
    def poison_scalar_selection(self, candidates, model, tokenizer):
        """选择最优投毒标量"""
        poisoned_sequences = {}
        scores = []
        
        # 准备目标样本
        target_inputs = self.prepare_target_sample_tensor(tokenizer)
        if target_inputs is None:
            return poisoned_sequences
        
        # 将目标样本移动到设备上
        inputs = {k: v.to(self.device) for k, v in target_inputs.items()}
        labels = inputs.pop('labels')
        
        # todo: 修复批量大小不匹配问题,这是大语言模型的训练，这样处理是不是不好
        max_len = min(inputs['input_ids'].size(1), labels.size(1))
        inputs['input_ids'] = inputs['input_ids'][:, :max_len]
        inputs['attention_mask'] = inputs['attention_mask'][:, :max_len]
        labels = labels[:, :max_len]
        
        # todo: 这里是根据候选种子查找到对应的种子标量对
        if isinstance(candidates, list):
            # 如果是列表，转换为字典
            candidates_dict = {}
            for s_i in candidates:
                if s_i in self.server.seed_pool:
                    candidates_dict[s_i] = self.server.seed_pool[s_i]
            candidates = candidates_dict
        
        # 对每个候选序列计算投毒标量
        for s_i, v_i in candidates.items():
            # 创建扰动模型
            eps = 1e-3
            model_pos = deepcopy(model).to(self.device)
            model_neg = deepcopy(model).to(self.device)
            
            # todo: 应用正负扰动 这里的方法错了，扰动模型的计算是原模型w加减s_i种子生成的扰动z_i乘eps，z_i和w的维度相同
            with torch.no_grad():
                for name, param in model_pos.named_parameters():
                    param.add_(eps * v_i)
                
                for name, param in model_neg.named_parameters():
                    param.add_(-eps * v_i)
            
            # 计算损失
            with torch.no_grad():
                inputs['labels'] = labels
                L_pos = model_pos(**inputs).loss.item()
                L_neg = model_neg(**inputs).loss.item()
                
                # 计算目标样本上的标量
                v_target = (L_pos - L_neg) / (2 * eps)
                
                # todo: 计算反应值：abs(v_target) / (abs(v_i) + abs(v_target))
                rho_i = abs(v_target) / abs(v_i) if v_i != 0 else float('inf')
                
                scores.append((rho_i, s_i, v_i, v_target))
        
        # 按反应值升序排序
        scores.sort(key=lambda x: x[0])
        
        # 选择top-k2个进行投毒
        for i in range(min(self.k2, len(scores))):
            rho, s, v, v_target = scores[i]
            # 投毒公式: v' = v - v_target/v
            v_poisoned = v - v_target / v if v != 0 else v
            poisoned_sequences[s] = v_poisoned
        
        return poisoned_sequences
    
    def record_client_response(self, client_idx, response):
        """记录客户端响应"""
        if client_idx not in self.client_responses:
            self.client_responses[client_idx] = []
        
        # 确保响应不为None
        if response is not None:
            self.client_responses[client_idx].append(response)
            print(f"已记录客户端 {client_idx} 的响应: {response}")
        else:
            print(f"警告: 客户端 {client_idx} 返回了空响应")
    
    def record_loss(self, client_idx, round_idx, model, tokenizer):
        """记录目标样本在当前模型上的损失"""
        if client_idx not in self.loss_history:
            self.loss_history[client_idx] = {}
        
        model.to(self.device)
        model.eval()
        
        # 构建输入文本
        instruction = self.target_sample.get("instruction", "")
        context = self.target_sample.get("context", "")
        response = self.target_sample.get("response", "")
        
        if context:
            input_text = f"Instruction: {instruction}\nContext: {context}\nResponse: "
        else:
            input_text = f"Instruction: {instruction}\nResponse: "
        
        full_text = input_text + response
        
        # 分词处理
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.args.max_length)
        
        # 创建标签，将非响应部分的标签设为-100（忽略）
        labels = inputs.input_ids.clone()
        
        # 计算响应开始的位置
        response_tokens = tokenizer(input_text, return_tensors="pt").input_ids[0]
        response_start = len(response_tokens)
        
        # 将非响应部分的标签设为-100
        labels[0, :response_start] = -100
        
        # 将输入和标签移动到设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs['labels'] = labels.to(self.device)
        
        # 计算损失
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.item()
        
        # 记录损失
        self.loss_history[client_idx][round_idx] = loss
        
        # 将模型移回CPU
        model.to('cpu')
        
        return loss
    
    def plot_loss_curve(self, client_idx, save_path):
        """绘制目标样本损失曲线"""
        if client_idx not in self.loss_history or not self.loss_history[client_idx]:
            print(f"Client {client_idx} has no loss records")
            return
        
        # 设置matplotlib后端为Agg，避免字体问题
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 关闭字体警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        plt.figure(figsize=(10, 6))
        
        rounds = list(self.loss_history[client_idx].keys())
        losses = list(self.loss_history[client_idx].values())
        
        plt.plot(rounds, losses, marker='o', linestyle='-', color='blue')
        
        # 使用英文标签
        plt.title(f'Loss Curve for Client {client_idx}')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Loss curve saved to {save_path}")
    
    # todo: 成员推理的方法需要优化
    def membership_inference(self, client_idx, T=5, delta=0.1, eta=0.6):
        """
        多轮投毒观察与成员身份推断
        """
        if client_idx not in self.client_responses:
            print(f"警告: 客户端 {client_idx} 没有响应记录")
            return "无法判断 (无响应记录)"
        
        responses = self.client_responses[client_idx]
        if len(responses) < 2:
            print(f"警告: 客户端 {client_idx} 的响应数量不足 ({len(responses)})")
            return f"无法判断 (响应数量不足: {len(responses)})"
        
        print(f"客户端 {client_idx} 有 {len(responses)} 条响应记录")
        
        # 检查损失历史
        if client_idx not in self.loss_history or len(self.loss_history[client_idx]) < 2:
            print(f"警告: 客户端 {client_idx} 的损失记录不足")
            return "无法判断 (损失记录不足)"
        
        # 获取损失历史
        rounds = sorted(self.loss_history[client_idx].keys())
        losses = [self.loss_history[client_idx][r] for r in rounds]
        
        # 计算损失波动
        count = 0
        prev_loss = losses[0]
        
        for i in range(1, len(losses)):
            current_loss = losses[i]
            
            # 检查是否为投毒轮次
            if i % self.poison_interval == 0:
                # 投毒轮次，检查损失是否显著增加
                if current_loss > prev_loss * (1 + delta):
                    count += 1
            else:
                # 非投毒轮次，检查损失波动
                fluctuation = abs(current_loss - prev_loss) / max(prev_loss, 1e-10)
                if fluctuation > delta:
                    count += 1
            
            prev_loss = current_loss
        
        # 计算置信度
        confidence = count / min(T, len(losses) - 1)
        
        if confidence > eta:
            return "Member"
        else:
            return "Non-member"
    
    def _compute_loss(self, response):
        """计算响应中的损失值"""
        if isinstance(response, dict) and 'avg_loss' in response:
            return response['avg_loss']
        elif isinstance(response, dict) and 'loss_history' in response:
            return sum(response['loss_history']) / len(response['loss_history']) if response['loss_history'] else 0
        else:
            return 0.0
    
    def poison_server_seed_pool(self, target_client_idx, current_round_seeds, tokenizer):
        """
        对服务器的种子池进行投毒
        
        参数:
            target_client_idx: 目标客户端索引
            current_round_seeds: 当前轮次使用的种子集合
            tokenizer: 分词器
            
        返回:
            被投毒的种子池
        """
        # 获取历史扰动序列
        history_perturbations = self.server.seed_pool
        
        # 计算相似度并获取候选投毒序列
        candidates = self.similarity_measurement(history_perturbations, current_round_seeds)
        
        # 选择最优投毒标量
        poisoned_sequences = self.poison_scalar_selection(candidates, self.server.model, tokenizer)
        
        # 创建投毒后的种子池副本
        poisoned_seed_pool = deepcopy(self.server.seed_pool)
        
        # 应用投毒
        for s, v_poisoned in poisoned_sequences.items():
            poisoned_seed_pool[s] = v_poisoned
        
        return poisoned_seed_pool 