import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_comparison(data, x_label, title, save_name):
    """
    绘制对比图
    """
    x = data[x_label]
    metrics = ['reward', 'success_rate', 'pot_rewards', 'p_sens', 'e_llm', 'p_tx', 'llm_accuracy', 'crb']
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=16)
    
    for idx, metric in enumerate(metrics):
        row = idx // 4
        col = idx % 4
        
        # 绘制柱状图
        x_pos = np.arange(len(x))
        width = 0.35
        
        axes[row, col].bar(x_pos - width/2, data[f'random_{metric}'], width, label='Random策略')
        axes[row, col].bar(x_pos + width/2, data[f'greedy_{metric}'], width, label='Greedy策略')
        
        # 设置标题和标签
        axes[row, col].set_title(f'{metric}对比')
        axes[row, col].set_xlabel(x_label)
        axes[row, col].set_ylabel('值')
        
        # 设置x轴刻度
        axes[row, col].set_xticks(x_pos)
        axes[row, col].set_xticklabels(x)
        
        # 添加图例
        axes[row, col].legend()
        
        # 添加网格
        axes[row, col].grid(True, linestyle='--', alpha=0.7)
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'/home/minrui/ISAC/{save_name}.png')
    plt.close()

def main():
    # 加载数据
    data_N = np.load('/home/minrui/ISAC/test_results_N.npy', allow_pickle=True).item()
    data_U = np.load('/home/minrui/ISAC/test_results_U.npy', allow_pickle=True).item()
    data_compute_units = np.load('/home/minrui/ISAC/test_results_compute_units.npy', allow_pickle=True).item()
    data_miner_rewards = np.load('/home/minrui/ISAC/test_results_miner_rewards.npy', allow_pickle=True).item()
    
    # 绘制对比图
    plot_comparison(data_N, 'N', '不同节点数量下的性能对比', 'comparison_N')
    plot_comparison(data_U, 'U', '不同目标数量下的性能对比', 'comparison_U')
    plot_comparison(data_compute_units, 'compute_units', '不同计算单元数下的性能对比', 'comparison_compute_units')
    plot_comparison(data_miner_rewards, 'miner_reward', '不同矿工奖励值下的性能对比', 'comparison_miner_rewards')
    
    print("可视化完成！图片已保存。")

if __name__ == "__main__":
    main() 