from optimizers.mezo_optimizer import *
from optimizers.mezo_bias_optimizer import *
from tqdm import tqdm


class Client(object):
    def __init__(self, idx, args, train_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None

        self.device = torch.device(f'cuda:{args.device}')

    def local_train_with_seed_pool(self, pulled_model, cur_round, selected_seeds, memory_record_dic=None, probabilities=None, gradient_history=None):
        self.model = pulled_model
        self.model.to(self.device)
        
        if memory_record_dic is not None:
            torch.cuda.empty_cache()
        
        # 只初始化选定种子的标量池 
        self.local_seed_pool = {seed: 0.0 for seed in selected_seeds}

        lr = self.args.lr
        
        # 修改迭代逻辑,确保完整训练本地数据
        if self.args.batch_or_epoch == 'epoch':
            # 训练指定轮数,每轮完整训练一遍数据
            num_epochs = self.args.local_step
        else:
            # 计算需要的epoch数,确保所有数据都被训练
            num_epochs = max(1, self.args.local_step // len(self.train_loader))
            
        if self.args.bias_sampling:
            framework = MeZOBiasOptimizer(self.model, args=self.args, lr=lr, candidate_seeds=selected_seeds, probabilities=probabilities, gradient_history=gradient_history)
        else:
            framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=selected_seeds)

        self.model.eval()
        with torch.inference_mode():
            for epoch in range(num_epochs):
                loss_total_train = 0.0
                num_trained = 0
                progress_bar = tqdm(range(len(self.train_loader)))
                self.train_iterator = iter(self.train_loader) # 每个epoch重置迭代器
                
                # 完整训练一遍数据集
                for _ in range(len(self.train_loader)):
                    batch = next(self.train_iterator)
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device), 
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    logits, loss = framework.zo_step(batch, local_seed_pool=self.local_seed_pool)
                    progress_bar.update(1)
                    
                    if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                        
                    progress_bar.set_description(f'client {self.idx} train at epoch {epoch+1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')

        # 释放内存
        del framework
        self.model = None
        
        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {}
            memory_record_dic[self.device.index]['max_memory_allocated'] = torch.cuda.max_memory_allocated(self.device)
            memory_record_dic[self.device.index]['max_memory_reserved'] = torch.cuda.max_memory_reserved(self.device)

    def clear_model(self):
        # clear model to same memory
        self.model = None

    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model