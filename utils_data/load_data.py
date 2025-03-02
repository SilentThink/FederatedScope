import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from utils_data.default_tokens import DefaultToken
from utils_data.partition_data import partition_idx_labeldir
from collections import Counter
import json


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)

    # Generation task
    if args.dataset == 'dolly':
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        if args.eval_metric == 'loss':
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        # only use a subset of raw dataset
        raw_datasets, _ = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * args.dataset_subsample), len(raw_datasets) - int(len(raw_datasets) * args.dataset_subsample)])
        y_all = np.array([item['categories'] for item in raw_datasets])
        index_eval = np.where(y_all == args.zerotask)[0]
        # delete the indices of eval samples from the all set
        index_train = np.delete(np.arange(len(y_all)), index_eval)
        raw_datasets = np.array(raw_datasets)
        train_set = raw_datasets[index_train]
        eval_set = raw_datasets[index_eval]
        y_train = np.array([item['categories'] for item in train_set])
        counter = Counter(y_train)
        noniid = args.iid
        if 'dir' in noniid:
            split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=float(noniid[3:]), num_classes=len(counter))
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_set, indices=sample_indices))
        else:
            n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_set) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            ) for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, collate_fn=data_collator
        )
        
    elif args.dataset in ['instruct']:
        from utils_data.natural_instruction_loader import get_instruction_dataset
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, eval_loader, tokenizer

def record_client_datasets(client_datasets, target_sample, output_file):
    """记录每个客户端的数据集中是否包含目标样本"""
    target_sample_json = json.loads(target_sample) if isinstance(target_sample, str) else target_sample
    
    # 检查每个客户端的数据集
    client_has_target = {}
    for client_idx, dataset in enumerate(client_datasets):
        has_target = False
        try:
            for item in dataset:
                # 比较指令和响应是否匹配
                if (item.get("instruction") == target_sample_json.get("instruction") and 
                    item.get("response") == target_sample_json.get("response")):
                    has_target = True
                    break
        except Exception as e:
            print(f"警告: 处理客户端 {client_idx} 数据集时出错: {str(e)}")
            has_target = False
            
        client_has_target[str(client_idx)] = has_target  # 确保键是字符串
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(client_has_target, f, indent=2)
    
    return client_has_target