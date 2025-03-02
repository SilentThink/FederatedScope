修改代码

'''
{"instruction": "Who are N-Dubz?", "context": "", "response": "N-Dubz are a popular band in the United Kingdom, made up of Tulisa, Fazer, and Dappy, formed out of London. The band were formed when they were young teenagers in the early 2000s. They were inspired to form the band by Dappy's late father, known to the band as \"Uncle B\". Their song \"Papa can you hear me?\" is a tribute to Uncle B. Tulisa and Dappy are cousins, whilst Fazer has always been a close friend.\nThe trio have had many successful hits, and collaborated with popular artists like Tinchy Strider and Skepta. They parted ways in 2011, and Dappy started a solo career, whilst Tulisa became a judge on the popular UK show \"The X Factor\". She formed and mentored the winning band Little Mix. \nThe band reunited in 2022 and released new music, along with a sold out UK tour.", "category": "general_qa"}
'''
需要记录客户端的数据集，判断攻击的目标数据是否在客户端i的本地数据集中。

攻击者针对databricks-dolly-15k.jsonl数据集中的上面这个数据实例，进行投毒攻击，判断该数据是否在客户端i的本地数据集中。

攻击者有对服务器端的控制权。每一大轮训练，服务器需要指定客户端训练的种子，而不是随机选择，同时如果进行攻击，服务器会对维护的全局种子标量池的部分进行篡改，首先筛选与该轮次指定客户端训练的种子重合或者对应的扰动向量相似的种子筛选k1个，之后再对这k1个种子根据优先级投毒标量选择进行筛选，筛选出k2个种子，之后对全局种子池中的这k2个种子的标量进行篡改。篡改后服务器再将初始模型加上篡改后的种子标量池更新的模型发给客户端进行本地训练。

客户端训练后将多轮的种子梯度标量上传给服务器，攻击者服务器根据目标客户端上传的多轮种子标量信息对全局种子标量池进行更新，进而得到多轮的全局模型，用这多轮模型对目标攻击数据进行评测，得到多轮损失数据。

修改设定，如果实施攻击，那么每轮参与训练的客户端就不是随机选，而是攻击的目标客户端一定参与训练，其他客户端随机选择

设定参数：投毒间隔T。每隔T轮服务器进行一次投毒，通过观察投毒后的T轮全局模型再目标数据的loss的变化判断是否是目标客户端的成员

将要攻击的数据保存再额外的文件中，同时要记录目标客户端训练集是否有该数据以判断最终攻击结果的正确性


