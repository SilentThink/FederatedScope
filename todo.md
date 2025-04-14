# 成员推理攻击实现要求

# （1）推理目标数据是否属于全体训练集
## 数据集准备
采用 databricks-dolly-15k.jsonl 数据集选取部分数据进行联邦训练
数据格式如下：
```
{
  "instruction": "指令",
  "context": "上下文",
  "response": "响应",
  "category": "类别"
}
{ 
  "instruction": "What is the Pizza history?", 
  "context": "Modern pizza evolved from similar flatbread dishes in Naples, Italy, in the 18th or early 19th century.\nThe word pizza was first documented in AD 997 in Gaeta and successively in different parts of Central and Southern Italy. Pizza was mainly eaten in Italy and by emigrants from there. This changed after World War II when Allied troops stationed in Italy came to enjoy pizza along with other Italian foods.", 
  "response": "The history of pizza begins in antiquity, as various ancient cultures produced basic flatbreads with several toppings.\n\nA precursor of pizza was probably the focaccia, a flatbread known to the Romans as panis focacius, to which toppings were then added. Modern pizza evolved from similar flatbread dishes in Naples, Italy, in the 18th or early 19th century.\n\nThe word pizza was first documented in AD 997 in Gaeta and successively in different parts of Central and Southern Italy. Pizza was mainly eaten in Italy and by emigrants from there. This changed after World War II when Allied troops stationed in Italy came to enjoy pizza along with other Italian foods.\nSome commentators have suggested that the origins of modern pizza can be traced to pizzarelle, which were kosher for Passover cookies eaten by Roman Jews after returning from the synagogue on that holiday, though some also trace its origins to other Italian paschal bread. Other examples of flatbreads that survive to this day from the ancient Mediterranean world are focaccia (which may date back as far as the ancient Etruscans); Manakish in the Levant, coca (which has sweet and savory varieties) from Catalonia, Valencia and the Balearic Islands; the Greek Pita; Lepinja in the Balkans; or Piadina in the Romagna part of Emilia-Romagna in Italy", 
  "category": "information_extraction"
}
```
将数据集中的数据按照比例划分,一部分用作训练集和测试集（再做划分），另一部分为nonmember数据集，从划分后的训练集以及nonmember数据集中选取成员推理攻击的目标数据


## 客户端训练数据划分
将训练集的数据划分给各个客户端

## 服务器端控制
- 攻击者拥有服务器端的控制权
服务器维持一个种子标量池，负责对全局模型的更新：全局模型(w) = 初始模型(w0) + sum(v_i*z_i)。其中v_i是种子标量池中的种子的梯度标量，z_i是由种子s_i生成的维度和w0相同的高斯向量。
- 每轮训练时：
  服务器选取特定比例的客户端参加本轮训练，每隔T轮进行一次投毒
  1. 服务器指定客户端训练的种子
  2. 如果进行攻击，服务器会篡改全局种子标量池：
    攻击者服务器选择本轮训练的候选种子集。计算当前全局模型在各个种子对应的梯度标量，在全局种子标量池中对应种子的标量篡改为$v_{i^{\ast}}^{j}{'} = v_{i^{\ast}}^{j} - v_{i^{\ast}}^{j}(x,y)/v_{i^{\ast}}^{j}$

  3. 将初始模型加上种子标量池更新的模型发给客户端

## 客户端训练
若客户端参与本轮训练，则会接收到服务器下发的当前全局模型和客户端本轮训练的候选种子
客户端利用本地训练集和候选种子对全局模型进行多轮本地微调训练
每次本地训练都在候选种子中抽取一个进行零阶优化微调，得到对应的梯度标量
- 客户端训练后上传多轮的种子梯度标量
- 服务器根据目标客户端上传的多轮种子标量信息更新全局种子标量池

## 聚合评估
- 服务器收到客户端上传的梯度标量后，进行聚合
- 聚合后，服务器将聚合后的梯度标量更新到全局模型
- 使用更新后的全局模型计算在各目标数据上的loss，并记录loss和对应的轮次

finished:
1.设计Attacker类，其继承Server，其作为攻击者在充当服务器的角色的同时可以选择在某个训练的轮次中对种子标量池进行投毒。

2.设计投毒成员推理攻击的实验。每隔5轮针对目标数据进行一次投毒，记录所有投毒和未投毒轮次得到的全局模型在目标数据上的loss。

3.每轮训练是再全体客户端中选择一定比例进行本地训练，但如果投毒轮次，目标数据中的成员数据并不在被选择的客户端数据集里，就没有对被篡改标量的修复现象。
解决方案：加大投毒间隔周期

todo:
4.在对当前轮次的种子投毒后，下一次训练的种子又要重新选取，客户端对投毒影响的修正只能在投毒本轮执行，这可能导致无法明显观察到修正作用
解决方案：如果投毒，在投毒周期的轮次中选取相同种子
python main.py --rounds 60 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset dolly --dataset_subsample 0.5 --num_clients 50 --lr 0.0000003 -K 2048 -m 0.2 --log --attack --attack_amplitude 0.5 --num_target 1 --target_member_ratio 1 --batch_or_epoch batch --local_step 200 --poison_interval 10
问题：只在投毒轮次有修复现象，之后的轮次loss反而越来越高了

5.种子的筛选:投毒后的轮次中使用同一候选种子集进行训练，不再重新选择

6.设计方案，根据投毒后loss的变化趋势进行成员判断

# （2）推理目标数据是否属于目标客户端的训练集

