import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_loss_pattern(df, target_id):
    """分析单个目标数据的loss变化模式"""
    target_data = df[df['Target_ID'] == target_id]
    
    # 获取投毒轮次
    poison_rounds = target_data[target_data['Is_Poison_Round'] == True]['Round'].values
    
    changes = []
    for poison_round in poison_rounds:
        if poison_round >= max(target_data['Round']):
            continue
            
        # 获取投毒轮次的loss和下一轮的loss
        poison_loss = target_data[target_data['Round'] == poison_round]['Target_Loss'].values[0]
        next_loss = target_data[target_data['Round'] == poison_round + 1]['Target_Loss'].values[0]
        
        # 计算恢复幅度 (loss的相对变化)
        recovery = abs(next_loss - poison_loss) / poison_loss
        changes.append(recovery)
    
    # 计算平均恢复幅度
    avg_change = np.mean(changes) if changes else 0
    return avg_change

def predict_membership(df):
    """预测所有目标数据的成员身份"""
    target_ids = df['Target_ID'].unique()
    recovery_patterns = {}
    
    # 分析每个目标的恢复模式
    for target_id in target_ids:
        recovery_rate = analyze_loss_pattern(df, target_id)
        recovery_patterns[target_id] = recovery_rate
    
    # 根据恢复幅度的分布确定阈值
    recoveries = list(recovery_patterns.values())
    threshold = np.mean(recoveries) + np.std(recoveries)
    
    # 预测成员身份
    predictions = {}
    for target_id, recovery in recovery_patterns.items():
        # 如果恢复幅度大，更可能是非成员
        predictions[target_id] = recovery < threshold
        
    return predictions, recovery_patterns

def visualize_results(df, predictions, recovery_patterns):
    """可视化分析结果"""
    plt.figure(figsize=(15, 10))
    
    # 绘制恢复幅度分布
    plt.subplot(2, 1, 1)
    member_recoveries = [r for id, r in recovery_patterns.items() if predictions[id]]
    non_member_recoveries = [r for id, r in recovery_patterns.items() if not predictions[id]]
    
    plt.hist([member_recoveries, non_member_recoveries], 
             label=['Predicted Members', 'Predicted Non-members'],
             bins=10)
    plt.xlabel('Recovery Rate')
    plt.ylabel('Count')
    plt.title('Distribution of Recovery Rates')
    plt.legend()
    
    # 保存结果
    plt.savefig('membership_inference_results.png')
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('logs/20250415_133129/results.csv')
    
    # 预测成员身份
    predictions, recovery_patterns = predict_membership(df)
    
    # 可视化结果
    visualize_results(df, predictions, recovery_patterns)
    
    # 保存预测结果
    results = pd.DataFrame({
        'Target_ID': list(predictions.keys()),
        'Is_Member': list(predictions.values()),
        'Recovery_Rate': [recovery_patterns[id] for id in predictions.keys()]
    })
    
    results = results.sort_values('Target_ID')
    results.to_csv('membership_inference_results.csv', index=False)
    
    # 打印结果统计
    print("\nMembership Inference Results:")
    print(f"Total targets: {len(predictions)}")
    print(f"Predicted members: {sum(predictions.values())}")
    print(f"Predicted non-members: {len(predictions) - sum(predictions.values())}")
    
    # 打印详细结果
    print("\nDetailed Results:")
    print(results)

if __name__ == "__main__":
    main()