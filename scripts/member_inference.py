import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_loss_pattern(df, target_id):
    """分析单个目标数据的loss变化特征"""
    target_data = df[df['Target_ID'] == target_id]
    poison_rounds = target_data[target_data['Is_Poison_Round'] == True]['Round'].values
    
    features = {
        'poison_increases': [], # 投毒轮次的上升幅度
        'recovery_drops': [],   # 投毒后的下降幅度
        'overall_trend': 0      # 整体下降趋势
    }
    
    # 计算投毒轮次的loss变化特征
    for poison_round in poison_rounds:
        if poison_round >= max(target_data['Round']):
            continue
            
        pre_loss = target_data[target_data['Round'] == poison_round - 1]['Target_Loss'].values[0]
        poison_loss = target_data[target_data['Round'] == poison_round]['Target_Loss'].values[0]
        next_loss = target_data[target_data['Round'] == poison_round + 1]['Target_Loss'].values[0]
        
        # 计算投毒导致的上升幅度
        poison_increase = (poison_loss - pre_loss) / pre_loss
        features['poison_increases'].append(poison_increase)
        
        # 计算投毒后的恢复下降幅度
        recovery_drop = (poison_loss - next_loss) / poison_loss
        features['recovery_drops'].append(recovery_drop)
    
    # 计算整体趋势(使用简单线性回归的斜率)
    rounds = target_data['Round'].values
    losses = target_data['Target_Loss'].values
    if len(rounds) > 1:
        slope = np.polyfit(rounds, losses, 1)[0]
        features['overall_trend'] = slope
    
    return features

def predict_membership(df):
    """基于单个目标loss变化特征判断成员身份，针对非成员场景优化"""
    target_ids = df['Target_ID'].unique()
    features_by_id = {}
    predictions = {}
    
    # 收集所有目标的特征
    for target_id in target_ids:
        features = analyze_loss_pattern(df, target_id)
        features_by_id[target_id] = features
    
    # 计算整体特征分布
    all_increases = []
    all_drops = []
    all_trends = []
    for features in features_by_id.values():
        if features['poison_increases']:
            all_increases.extend(features['poison_increases'])
        if features['recovery_drops']:
            all_drops.extend(features['recovery_drops'])
        all_trends.append(features['overall_trend'])
    
    # 针对非成员特征优化判定标准
    for target_id, features in features_by_id.items():
        # 1. 投毒效果不显著（对非成员影响较小）
        has_low_poison_effect = True
        if features['poison_increases']:
            avg_increase = np.mean(features['poison_increases'])
            has_low_poison_effect = avg_increase < 0.05  # 提高阈值
            
        # 2. 恢复现象不明显
        has_low_recovery = True
        if features['recovery_drops']:
            avg_drop = np.mean(features['recovery_drops'])
            has_low_recovery = avg_drop < 0.03  # 提高阈值
            
        # 3. 总体趋势较为平稳
        has_stable_trend = abs(features['overall_trend']) < 0.001  # 更严格的趋势要求
            
        # 综合判断：满足所有条件才认为是非成员
        predictions[target_id] = not (has_low_poison_effect and 
                                    has_low_recovery and 
                                    has_stable_trend)
        
    return predictions, features_by_id

def main():
    # 读取数据
    df = pd.read_csv('../logs/20250415_144439(no)/results.csv')
    
    # 预测成员身份
    predictions, features_by_id = predict_membership(df)
    
    # 保存预测结果
    results = pd.DataFrame({
        'Target_ID': list(predictions.keys()),
        'Is_Member': list(predictions.values()),
        'Poison_Increase': [np.mean(features_by_id[id]['poison_increases']) if features_by_id[id]['poison_increases'] else np.nan for id in predictions.keys()],
        'Recovery_Drop': [np.mean(features_by_id[id]['recovery_drops']) if features_by_id[id]['recovery_drops'] else np.nan for id in predictions.keys()],
        'Overall_Trend': [features_by_id[id]['overall_trend'] for id in predictions.keys()]
    })
    
    results = results.sort_values('Target_ID')
    results.to_csv('membership_inference_results.csv', index=False)
    
    # 打印结果统计
    print("\n成员推理结果:")
    print(f"总目标数量: {len(predictions)}")
    print(f"预测为成员数量: {sum(predictions.values())}")
    print(f"预测为非成员数量: {len(predictions) - sum(predictions.values())}")
    print(f"预测成员比例: {sum(predictions.values())/len(predictions)*100:.2f}%")
    
    # 打印详细结果
    print("\n详细结果:")
    print(results[['Target_ID', 'Is_Member']].to_string())

if __name__ == "__main__":
    main()