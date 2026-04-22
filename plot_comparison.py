import numpy as np
import matplotlib.pyplot as plt
from math import pi

plt.rcParams.update({
    "text.usetex": False,              # 若支持 LaTeX 可设为 True
    "font.family": "serif",            # 字体为衬线体，符合IEEE期刊风格
    "font.size": 12,                    # 字号通常为8pt或9pt
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.figsize": [3.5, 2.8],      # 调整上方空间，减少高度
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "figure.subplot.top": 0.95,        # 调整顶部空白
    "figure.subplot.bottom": 0.21,      # 调整底部空白
    "figure.subplot.left": 0.21,        # 调整左侧空白
    "figure.subplot.right": 0.95,      # 调整右侧空白
})

# 美观配色方案（Color Universal Design, CUD）
colors = [
    "#E69F00",  # orange
    "#56B4E9",  # blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # dark blue
    "#D55E00",  # reddish orange
    "#CC79A7",  # purple
]
metrics = ['reward', 'success_rate', 'pot_rewards', 'p_sens', 'e_llm', 'p_tx', 'llm_accuracy', 'crb']
    
# 这个是用来画对比图的, 我们使用柱状图来画不同算法在不同环境设置下的表现
data_N = np.load('/home/minrui/ISAC/test_results_N.npy', allow_pickle=True).item()
data_U = np.load('/home/minrui/ISAC/test_results_U.npy', allow_pickle=True).item()
data_compute_units = np.load('/home/minrui/ISAC/test_results_compute_units.npy', allow_pickle=True).item()
data_miner_rewards = np.load('/home/minrui/ISAC/test_results_miner_rewards.npy', allow_pickle=True).item()

# 对比不同环境设置下的表现
metrics = ['reward', 'success_rate', 'pot_rewards', 'p_sens', 'e_llm', 'p_tx', 'llm_accuracy', 'crb']

random_N = {metric: data_N[f'random_{metric}'] for metric in metrics}
random_U = {metric: data_U[f'random_{metric}'] for metric in metrics}
random_compute_units = {metric: data_compute_units[f'random_{metric}'] for metric in metrics}
random_miner_rewards = {metric: data_miner_rewards[f'random_{metric}'] for metric in metrics}

greedy_N = {metric: data_N[f'greedy_{metric}'] for metric in metrics}
greedy_U = {metric: data_U[f'greedy_{metric}'] for metric in metrics}
greedy_compute_units = {metric: data_compute_units[f'greedy_{metric}'] for metric in metrics}
greedy_miner_rewards = {metric: data_miner_rewards[f'greedy_{metric}'] for metric in metrics}

PPO_reward_N =  [0.7079580366611481, 2.2253363633155825, 2.543781008720398, 3.1746511220932008]
PPO_success_rate_N =  [0.39, 0.38, 0.38, 0.36]
PPO_pot_rewards_N =  [1.3539792, 1.3829933, 1.4311552, 1.4058485]
PPO_p_sens_N =  [0.33821663, 0.33850682, 0.37071866, 0.3548493]
PPO_e_llm_N =  [0.33765244, 0.1784352, 0.20411143, 0.3049056]
PPO_p_tx_N =  [0.32413077, 0.124272525, 0.2203799, 0.11116337]
PPO_llm_accuracy_N =  [0.88619554, 0.83739823, 0.83552325, 0.84959227]
PPO_crb_N =  [5.32539, 4.9244704, 4.9263444, 4.8870535]

PPO_reward_U =  [3.5114367210865023, 2.2253363633155825, 1.1918999755382538, -0.7773425889015197, -0.04945501208305359]
PPO_success_rate_U =  [0.5, 0.38, 0.3, 0.19, 0.19]
PPO_pot_rewards_U =  [1.836979, 1.3829933, 1.09961, 0.7212507, 0.68378264]
PPO_p_sens_U =  [0.2643701, 0.33850682, 0.28947067, 0.33633187, 0.3272533]
PPO_e_llm_U =  [0.28229973, 0.1784352, 0.30455267, 0.47266296, 0.19601005]
PPO_p_tx_U =  [0.11983036, 0.124272525, 0.108286716, 0.17136998, 0.1770044]
PPO_llm_accuracy_U =  [0.8568377, 0.83739823, 0.8497864, 0.8826094, 0.84136796]
PPO_crb_U =  [4.9039836, 4.9244704, 5.4827905, 4.9459834, 4.755098]

PPO_reward_compute_units =  [1.432046805024147, 1.3698640441894532, 2.2253363633155825, 2.207975358963013, 2.30570465028286]
PPO_success_rate_compute_units =  [0.27, 0.27, 0.38, 0.38, 0.38]
PPO_pot_rewards_compute_units =  [0.9580933, 0.96198344, 1.3829933, 1.4062003, 1.3669494]
PPO_p_sens_compute_units =  [0.28256848, 0.28742415, 0.33850682, 0.32711163, 0.36954427]
PPO_e_llm_compute_units =  [0.14894038, 0.1612773, 0.1784352, 0.3016737, 0.17323609]
PPO_p_tx_compute_units =  [0.049235526, 0.056660656, 0.124272525, 0.04142325, 0.055600878]
PPO_llm_accuracy_compute_units =  [0.81623316, 0.81979614, 0.83739823, 0.85764676, 0.8314351]
PPO_crb_compute_units =  [5.2905383, 5.0897074, 4.9244704, 4.759791, 5.025035]

PPO_reward_miner_rewards =  [2.049576816558838, 2.110243582725525, 2.2253363633155825, 2.4059888744354248, 2.544029983282089]
PPO_success_rate_miner_rewards =  [0.38, 0.38, 0.38, 0.38, 0.38]
PPO_pot_rewards_miner_rewards =  [1.2407833, 1.3121353, 1.3829933, 1.4401267, 1.5151155]
PPO_p_sens_miner_rewards =  [0.31153175, 0.3222467, 0.33850682, 0.38048768, 0.31602865]
PPO_e_llm_miner_rewards =  [0.13857278, 0.16433068, 0.1784352, 0.16329746, 0.24058466]
PPO_p_tx_miner_rewards =  [0.1074866, 0.12214327, 0.124272525, 0.094345294, 0.1104921]
PPO_llm_accuracy_miner_rewards =  [0.83127105, 0.8358173, 0.83739823, 0.8348988, 0.853054]
PPO_crb_miner_rewards =  [5.0993643, 5.027281, 4.9244704, 4.9365745, 4.8420653]

# 定义N的取值
N_values = [2, 3, 4, 5]
U_values = [4, 5, 6, 7, 8]
compute_units_values = [8, 10, 12, 14, 16]
miner_rewards_values = [0.3, 0.4, 0.5, 0.6, 0.7]

# 获取对应的reward值
random_rewards_N = [random_N['reward'][i] for i in range(len(N_values))]
greedy_rewards_N = [greedy_N['reward'][i] for i in range(len(N_values))]
PPO_rewards_N = PPO_reward_N

random_rewards_U = [random_U['reward'][i] for i in range(len(U_values))]
greedy_rewards_U = [greedy_U['reward'][i] for i in range(len(U_values))]
PPO_rewards_U = PPO_reward_U

random_rewards_compute_units = [random_compute_units['reward'][i] for i in range(len(compute_units_values))]
greedy_rewards_compute_units = [greedy_compute_units['reward'][i] for i in range(len(compute_units_values))]
PPO_rewards_compute_units = PPO_reward_compute_units

random_rewards_miner_rewards = [random_miner_rewards['reward'][i] for i in range(len(miner_rewards_values))]
greedy_rewards_miner_rewards = [greedy_miner_rewards['reward'][i] for i in range(len(miner_rewards_values))]
PPO_rewards_miner_rewards = PPO_reward_miner_rewards

# 画N的折线图
plt.figure()
plt.plot(N_values, random_rewards_N, marker='o', label='Random', color=colors[0])
plt.plot(N_values, greedy_rewards_N, marker='o', label='Greedy', color=colors[1])
plt.plot(N_values, PPO_rewards_N, marker='o', label='PPO', color=colors[2])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(N_values, [int(x) for x in N_values])
plt.xlabel('N')
plt.ylabel('Reward')
plt.legend()
plt.savefig('/home/minrui/ISAC/compare_reward_N.pdf')

# 画U的折线图
plt.figure()
plt.plot(U_values, random_rewards_U, marker='o', label='Random', color=colors[0])
plt.plot(U_values, greedy_rewards_U, marker='o', label='Greedy', color=colors[1])
plt.plot(U_values, PPO_rewards_U, marker='o', label='PPO', color=colors[2])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(U_values, [int(x) for x in U_values])
plt.xlabel('U')
plt.ylabel('Reward')
plt.legend()
plt.savefig('/home/minrui/ISAC/compare_reward_U.pdf')

# 画compute_units的折线图
plt.figure()
plt.plot(compute_units_values, random_rewards_compute_units, marker='o', label='Random', color=colors[0])
plt.plot(compute_units_values, greedy_rewards_compute_units, marker='o', label='Greedy', color=colors[1])
plt.plot(compute_units_values, PPO_rewards_compute_units, marker='o', label='PPO', color=colors[2])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(compute_units_values, [int(x) for x in compute_units_values])
plt.xlabel('Compute Units')
plt.ylabel('Reward')
plt.legend()
plt.savefig('/home/minrui/ISAC/compare_reward_compute_units.pdf')

# 画miner_rewards的折线图
plt.figure()
plt.plot(miner_rewards_values, random_rewards_miner_rewards, marker='o', label='Random', color=colors[0])
plt.plot(miner_rewards_values, greedy_rewards_miner_rewards, marker='o', label='Greedy', color=colors[1])
plt.plot(miner_rewards_values, PPO_rewards_miner_rewards, marker='o', label='PPO', color=colors[2])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(miner_rewards_values, [float(x) for x in miner_rewards_values])
plt.xlabel('Miner Rewards')
plt.ylabel('Reward')
plt.legend()
plt.savefig('/home/minrui/ISAC/compare_reward_miner_rewards.pdf')

# 用柱状图对比各个算法在不同设置下的success_rate

# 获取对应的reward值
random_success_rate_N = [random_N['success_rate'][i] for i in range(len(N_values))]
greedy_success_rate_N = [greedy_N['success_rate'][i] for i in range(len(N_values))]
PPO_success_rate_N = PPO_success_rate_N

random_success_rate_U = [random_U['success_rate'][i] for i in range(len(U_values))]
greedy_success_rate_U = [greedy_U['success_rate'][i] for i in range(len(U_values))]
PPO_success_rate_U = PPO_success_rate_U

random_success_rate_compute_units = [random_compute_units['success_rate'][i] for i in range(len(compute_units_values))]
greedy_success_rate_compute_units = [greedy_compute_units['success_rate'][i] for i in range(len(compute_units_values))]
PPO_success_rate_compute_units = PPO_success_rate_compute_units

random_success_rate_miner_rewards = [random_miner_rewards['success_rate'][i] for i in range(len(miner_rewards_values))]
greedy_success_rate_miner_rewards = [greedy_miner_rewards['success_rate'][i] for i in range(len(miner_rewards_values))]
PPO_success_rate_miner_rewards = PPO_success_rate_miner_rewards

# 画N的柱状图
plt.figure()
bar_width = 0.25
index = np.arange(len(N_values))

plt.bar(index, random_success_rate_N, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_success_rate_N, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_success_rate_N, bar_width, label='PPO', color=colors[2])

plt.xlabel('N')
plt.ylabel('Success Rate')
plt.xticks(index + bar_width, [int(x) for x in N_values])
plt.legend()
plt.ylim(0, 0.6)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_success_rate_N.pdf')

# 画U的柱状图
plt.figure()
index = np.arange(len(U_values))

plt.bar(index, random_success_rate_U, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_success_rate_U, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_success_rate_U, bar_width, label='PPO', color=colors[2])

plt.xlabel('U')
plt.ylabel('Success Rate')
plt.xticks(index + bar_width, [int(x) for x in U_values])
plt.legend()
plt.ylim(0, 0.6)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_success_rate_U.pdf')

# 画compute_units的柱状图
plt.figure()
index = np.arange(len(compute_units_values))

plt.bar(index, random_success_rate_compute_units, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_success_rate_compute_units, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_success_rate_compute_units, bar_width, label='PPO', color=colors[2])

plt.xlabel('Compute Units')
plt.ylabel('Success Rate')
plt.xticks(index + bar_width, [int(x) for x in compute_units_values])
plt.legend()
plt.ylim(0, 0.6)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_success_rate_compute_units.pdf')

# 画miner_rewards的柱状图
plt.figure()
index = np.arange(len(miner_rewards_values))

plt.bar(index, random_success_rate_miner_rewards, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_success_rate_miner_rewards, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_success_rate_miner_rewards, bar_width, label='PPO', color=colors[2])

plt.xlabel('Miner Rewards')
plt.ylabel('Success Rate')
plt.xticks(index + bar_width, [float(x) for x in miner_rewards_values])
plt.legend()
plt.ylim(0, 0.6)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_success_rate_miner_rewards.pdf')

# 计算各个算法在不同设置下的cost
random_cost_N = [random_N['p_sens'][i] + random_N['e_llm'][i] + random_N['p_tx'][i] for i in range(len(N_values))]
greedy_cost_N = [greedy_N['p_sens'][i] + greedy_N['e_llm'][i] + greedy_N['p_tx'][i] for i in range(len(N_values))]
PPO_cost_N = [PPO_p_sens_N[i] + PPO_e_llm_N[i] + PPO_p_tx_N[i] for i in range(len(N_values))]

random_cost_U = [random_U['p_sens'][i] + random_U['e_llm'][i] + random_U['p_tx'][i] for i in range(len(U_values))]
greedy_cost_U = [greedy_U['p_sens'][i] + greedy_U['e_llm'][i] + greedy_U['p_tx'][i] for i in range(len(U_values))]
PPO_cost_U = [PPO_p_sens_U[i] + PPO_e_llm_U[i] + PPO_p_tx_U[i] for i in range(len(U_values))]

random_cost_compute_units = [random_compute_units['p_sens'][i] + random_compute_units['e_llm'][i] + random_compute_units['p_tx'][i] for i in range(len(compute_units_values))]
greedy_cost_compute_units = [greedy_compute_units['p_sens'][i] + greedy_compute_units['e_llm'][i] + greedy_compute_units['p_tx'][i] for i in range(len(compute_units_values))]
PPO_cost_compute_units = [PPO_p_sens_compute_units[i] + PPO_e_llm_compute_units[i] + PPO_p_tx_compute_units[i] for i in range(len(compute_units_values))]

random_cost_miner_rewards = [random_miner_rewards['p_sens'][i] + random_miner_rewards['e_llm'][i] + random_miner_rewards['p_tx'][i] for i in range(len(miner_rewards_values))]
greedy_cost_miner_rewards = [greedy_miner_rewards['p_sens'][i] + greedy_miner_rewards['e_llm'][i] + greedy_miner_rewards['p_tx'][i] for i in range(len(miner_rewards_values))]
PPO_cost_miner_rewards = [PPO_p_sens_miner_rewards[i] + PPO_e_llm_miner_rewards[i] + PPO_p_tx_miner_rewards[i] for i in range(len(miner_rewards_values))]

# 提取各个算法在不同设置下的llm_accuracy和crb
random_llm_accuracy_N = [random_N['llm_accuracy'][i] for i in range(len(N_values))]
greedy_llm_accuracy_N = [greedy_N['llm_accuracy'][i] for i in range(len(N_values))]
PPO_llm_accuracy_N = PPO_llm_accuracy_N

random_crb_N = [random_N['crb'][i] for i in range(len(N_values))]
greedy_crb_N = [greedy_N['crb'][i] for i in range(len(N_values))]
PPO_crb_N = PPO_crb_N

random_llm_accuracy_U = [random_U['llm_accuracy'][i] for i in range(len(U_values))]
greedy_llm_accuracy_U = [greedy_U['llm_accuracy'][i] for i in range(len(U_values))]
PPO_llm_accuracy_U = PPO_llm_accuracy_U

random_crb_U = [random_U['crb'][i] for i in range(len(U_values))]
greedy_crb_U = [greedy_U['crb'][i] for i in range(len(U_values))]
PPO_crb_U = PPO_crb_U

random_llm_accuracy_compute_units = [random_compute_units['llm_accuracy'][i] for i in range(len(compute_units_values))]
greedy_llm_accuracy_compute_units = [greedy_compute_units['llm_accuracy'][i] for i in range(len(compute_units_values))]
PPO_llm_accuracy_compute_units = PPO_llm_accuracy_compute_units

random_crb_compute_units = [random_compute_units['crb'][i] for i in range(len(compute_units_values))]
greedy_crb_compute_units = [greedy_compute_units['crb'][i] for i in range(len(compute_units_values))]
PPO_crb_compute_units = PPO_crb_compute_units

random_llm_accuracy_miner_rewards = [random_miner_rewards['llm_accuracy'][i] for i in range(len(miner_rewards_values))]
greedy_llm_accuracy_miner_rewards = [greedy_miner_rewards['llm_accuracy'][i] for i in range(len(miner_rewards_values))]
PPO_llm_accuracy_miner_rewards = PPO_llm_accuracy_miner_rewards

random_crb_miner_rewards = [random_miner_rewards['crb'][i] for i in range(len(miner_rewards_values))]
greedy_crb_miner_rewards = [greedy_miner_rewards['crb'][i] for i in range(len(miner_rewards_values))]
PPO_crb_miner_rewards = PPO_crb_miner_rewards

# 画N的柱状图
plt.figure()
bar_width = 0.25
index = np.arange(len(N_values))

plt.bar(index, random_cost_N, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_cost_N, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_cost_N, bar_width, label='PPO', color=colors[2])

plt.xlabel('N')
plt.ylabel('Cost')
plt.xticks(index + bar_width, [int(x) for x in N_values])
plt.legend()
plt.ylim(0, max(max(random_cost_N), max(greedy_cost_N), max(PPO_cost_N)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_cost_N.pdf')

# 画U的柱状图
plt.figure()
index = np.arange(len(U_values))

plt.bar(index, random_cost_U, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_cost_U, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_cost_U, bar_width, label='PPO', color=colors[2])

plt.xlabel('U')
plt.ylabel('Cost')
plt.xticks(index + bar_width, [int(x) for x in U_values])
plt.legend()
plt.ylim(0, max(max(random_cost_U), max(greedy_cost_U), max(PPO_cost_U)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_cost_U.pdf')

# 画compute_units的柱状图
plt.figure()
index = np.arange(len(compute_units_values))

plt.bar(index, random_cost_compute_units, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_cost_compute_units, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_cost_compute_units, bar_width, label='PPO', color=colors[2])

plt.xlabel('Compute Units')
plt.ylabel('Cost')
plt.xticks(index + bar_width, [int(x) for x in compute_units_values])
plt.legend()
plt.ylim(0, max(max(random_cost_compute_units), max(greedy_cost_compute_units), max(PPO_cost_compute_units)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_cost_compute_units.pdf')

# 画miner_rewards的柱状图
plt.figure()
index = np.arange(len(miner_rewards_values))

plt.bar(index, random_cost_miner_rewards, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_cost_miner_rewards, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_cost_miner_rewards, bar_width, label='PPO', color=colors[2])

plt.xlabel('Miner Rewards')
plt.ylabel('Cost')
plt.xticks(index + bar_width, [float(x) for x in miner_rewards_values])
plt.legend()
plt.ylim(0, max(max(random_cost_miner_rewards), max(greedy_cost_miner_rewards), max(PPO_cost_miner_rewards)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_cost_miner_rewards.pdf')

# 画N的柱状图
plt.figure()
bar_width = 0.25
index = np.arange(len(N_values))

plt.bar(index, random_llm_accuracy_N, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_llm_accuracy_N, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_llm_accuracy_N, bar_width, label='PPO', color=colors[2])

plt.xlabel('N')
plt.ylabel('LLM Accuracy')
plt.xticks(index + bar_width, [int(x) for x in N_values])
plt.legend()
plt.ylim(0, max(max(random_llm_accuracy_N), max(greedy_llm_accuracy_N), max(PPO_llm_accuracy_N)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_llm_accuracy_N.pdf')

# 画U的柱状图
plt.figure()
index = np.arange(len(U_values))

plt.bar(index, random_llm_accuracy_U, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_llm_accuracy_U, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_llm_accuracy_U, bar_width, label='PPO', color=colors[2])

plt.xlabel('U')
plt.ylabel('LLM Accuracy')
plt.xticks(index + bar_width, [int(x) for x in U_values])
plt.legend()
plt.ylim(0, max(max(random_llm_accuracy_U), max(greedy_llm_accuracy_U), max(PPO_llm_accuracy_U)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_llm_accuracy_U.pdf')

# 画compute_units的柱状图
plt.figure()
index = np.arange(len(compute_units_values))

plt.bar(index, random_llm_accuracy_compute_units, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_llm_accuracy_compute_units, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_llm_accuracy_compute_units, bar_width, label='PPO', color=colors[2])

plt.xlabel('Compute Units')
plt.ylabel('LLM Accuracy')
plt.xticks(index + bar_width, [int(x) for x in compute_units_values])
plt.legend()
plt.ylim(0, max(max(random_llm_accuracy_compute_units), max(greedy_llm_accuracy_compute_units), max(PPO_llm_accuracy_compute_units)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_llm_accuracy_compute_units.pdf')

# 画miner_rewards的柱状图
plt.figure()
index = np.arange(len(miner_rewards_values))

plt.bar(index, random_llm_accuracy_miner_rewards, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_llm_accuracy_miner_rewards, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_llm_accuracy_miner_rewards, bar_width, label='PPO', color=colors[2])

plt.xlabel('Miner Rewards')
plt.ylabel('LLM Accuracy')
plt.xticks(index + bar_width, [float(x) for x in miner_rewards_values])
plt.legend()
plt.ylim(0, max(max(random_llm_accuracy_miner_rewards), max(greedy_llm_accuracy_miner_rewards), max(PPO_llm_accuracy_miner_rewards)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_llm_accuracy_miner_rewards.pdf')

# 画N的柱状图
plt.figure()
bar_width = 0.25
index = np.arange(len(N_values))

plt.bar(index, random_crb_N, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_crb_N, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_crb_N, bar_width, label='PPO', color=colors[2])

plt.xlabel('N')
plt.ylabel('SINR')
plt.xticks(index + bar_width, [int(x) for x in N_values])
plt.legend()
plt.ylim(0, max(max(random_crb_N), max(greedy_crb_N), max(PPO_crb_N)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_crb_N.pdf')

# 画U的柱状图
plt.figure()
index = np.arange(len(U_values))

plt.bar(index, random_crb_U, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_crb_U, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_crb_U, bar_width, label='PPO', color=colors[2])

plt.xlabel('U')
plt.ylabel('SINR')
plt.xticks(index + bar_width, [int(x) for x in U_values])
plt.legend()
plt.ylim(0, max(max(random_crb_U), max(greedy_crb_U), max(PPO_crb_U)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_crb_U.pdf')

# 画compute_units的柱状图
plt.figure()
index = np.arange(len(compute_units_values))

plt.bar(index, random_crb_compute_units, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_crb_compute_units, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_crb_compute_units, bar_width, label='PPO', color=colors[2])

plt.xlabel('Compute Units')
plt.ylabel('SINR')
plt.xticks(index + bar_width, [int(x) for x in compute_units_values])
plt.legend()
plt.ylim(0, max(max(random_crb_compute_units), max(greedy_crb_compute_units), max(PPO_crb_compute_units)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_crb_compute_units.pdf')

# 画miner_rewards的柱状图
plt.figure()
index = np.arange(len(miner_rewards_values))

plt.bar(index, random_crb_miner_rewards, bar_width, label='Random', color=colors[0])
plt.bar(index + bar_width, greedy_crb_miner_rewards, bar_width, label='Greedy', color=colors[1])
plt.bar(index + 2 * bar_width, PPO_crb_miner_rewards, bar_width, label='PPO', color=colors[2])

plt.xlabel('Miner Rewards')
plt.ylabel('SINR')
plt.xticks(index + bar_width, [float(x) for x in miner_rewards_values])
plt.legend()
plt.ylim(0, max(max(random_crb_miner_rewards), max(greedy_crb_miner_rewards), max(PPO_crb_miner_rewards)) * 1.4)  # 调整y轴的留白
plt.savefig('/home/minrui/ISAC/compare_crb_miner_rewards.pdf')