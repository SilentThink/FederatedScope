import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn')
sns.set_style("whitegrid")

# Read CSV file
df = pd.read_csv('../logs/20250415_135008/results.csv')

# Calculate average Target_Loss for each Round
avg_loss = df.groupby('Round')[['Target_Loss', 'Is_Poison_Round']].agg({
    'Target_Loss': 'mean',
    'Is_Poison_Round': 'first'
}).reset_index()

# 打印出数据检查
print("数据类型:")
print(avg_loss.dtypes)
print("\n投毒轮次数据:")
print(avg_loss[['Round', 'Is_Poison_Round']])

# Create figure
plt.figure(figsize=(10, 6))

# Plot average loss curve (z-order=1表示在底层)
plt.plot(avg_loss['Round'], avg_loss['Target_Loss'], 
         marker='o', linewidth=2, markersize=6, label='Average Target Loss',
         zorder=1)

# Mark poison rounds with larger markers (z-order=2表示在上层)
poison_rounds = avg_loss[avg_loss['Is_Poison_Round'].astype(str).str.lower() == 'true']
plt.scatter(poison_rounds['Round'], poison_rounds['Target_Loss'], 
           color='red', s=200, marker='*', label='Poison Round',
           zorder=2)

# Set plot properties
plt.title('Target Loss vs Training Rounds', fontsize=14)
plt.xlabel('Training Round', fontsize=12)
plt.ylabel('Average Target Loss', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, zorder=0)  # 网格放在最底层

# Save plot
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()