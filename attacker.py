from server import Server
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from optimizers.mezo_optimizer import MeZOFramework


class Attacker(Server):
    def __init__(self, args, eval_loader, candidate_seeds, log_dir, target_loader, poison_interval=5):
        """
        Args:
            args: 全局参数配置
            eval_loader: 评估数据加载器
            candidate_seeds: 候选种子集合
            log_dir: 日志目录
            target_loader: 目标数据加载器
            poison_interval: 投毒间隔轮次，默认每5轮进行一次投毒
        """
        super().__init__(args, eval_loader, candidate_seeds, log_dir)
        self.target_loader = target_loader  # 攻击目标数据
        self.poison_interval = poison_interval  # 投毒间隔
        self.poison_rounds = []  # 记录投毒轮次
        self.target_loss_history = {idx: [] for idx in range(len(target_loader.dataset))}  # 记录每个目标数据的loss历史
        self.attack_amplitude = args.attack_amplitude  # 攻击幅度
        
    def should_poison(self, round):
        """判断当前轮次是否需要投毒"""
        return round % self.poison_interval == 0
    
    def record_target_loss(self, round):
        """记录所有目标样本的损失（每轮执行）"""
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.inference_mode():
            for idx, batch in enumerate(self.target_loader):
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device) 
                }
                outputs = self.model(**batch)
                loss = outputs.loss.item()
                self.target_loss_history[idx].append((round, loss))
        
        self.model = self.model.cpu()

    def poison_seed_pool(self, round):
        """对种子标量池进行投毒"""
        if not self.should_poison(round=round):
            return
        
        print("Attacker is poisoning seed pool...")
        self.poison_rounds.append(round)
        
        # 计算衰减系数 (round从0开始)
        decay = 1.0 / (1.0 + round/self.args.rounds)  # 随轮数增加而衰减
        current_amplitude = self.attack_amplitude * decay
        
        # 将当前模型转移到GPU
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.inference_mode():
            for idx, batch in enumerate(self.target_loader):
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device) 
                }
                # 对种子标量池进行投毒
                for seed in self.selected_seeds:
                    # 调用mezo计算目标样本在当前种子下的梯度标量
                    framework = MeZOFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=[seed])
                    
                    # 计算目标样本在当前种子下的梯度标量
                    framework._zo_perturb_parameters(scaling_factor=1)
                    _, loss1 = framework.zo_forward(batch)
                    
                    framework._zo_perturb_parameters(scaling_factor=-2) 
                    _, loss2 = framework.zo_forward(batch)
                    
                    # 重置模型参数
                    framework._zo_perturb_parameters(scaling_factor=1)
                    
                    # 计算目标样本的梯度标量
                    v_target = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                    
                    # 篡改种子标量池中对应种子的标量
                    if seed in self.seed_pool:
                        # 使用衰减后的攻击幅度
                        self.seed_pool[seed] -= current_amplitude * v_target
                    
                    del framework

        # 投毒完成后将模型移回CPU以节省显存
        self.model = self.model.cpu()
    
    def select_seeds_for_round(self):
        """重写服务器的种子选择方法
        - 一半种子从上一轮攻击使用的种子中选择恢复效果最大的
        - 另一半从剩余种子中随机选择
        """
        num_seeds_per_part = self.args.num_seed // 2  # 每部分选择的种子数量
        
        if not hasattr(self, 'last_attack_seeds') or not self.last_attack_seeds:
            # 首次选择或没有历史攻击种子时,完全随机选择
            self.selected_seeds = np.random.choice(self.candidate_seeds, 
                                                 size=self.args.num_seed, 
                                                 replace=False)
            return
            
        # 计算上轮攻击种子的恢复效果(使用种子对应的梯度标量绝对值)
        recovery_effects = {}
        for seed in self.last_attack_seeds:
            if seed in self.seed_pool:
                recovery_effects[seed] = abs(self.seed_pool[seed])
            
        # 按恢复效果排序并选择效果最好的一半种子
        sorted_seeds = sorted(recovery_effects.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        selected_old = [seed for seed, _ in sorted_seeds[:num_seeds_per_part]]
        
        # 从剩余种子中随机选择另一半
        remaining_seeds = list(set(self.candidate_seeds) - set(selected_old))
        selected_new = np.random.choice(remaining_seeds,
                                      size=self.args.num_seed - len(selected_old),
                                      replace=False)
        
        self.selected_seeds = np.concatenate([selected_old, selected_new])
        
        # 保存本轮选择的种子用于下一轮参考
        if self.should_poison(round=self.current_round):
            self.last_attack_seeds = self.selected_seeds.copy()


