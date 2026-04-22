import matplotlib.pyplot as plt
import re

# 读取文件内容
file_path = '/home/minrui/ISAC/logN2.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取奖励数据
rewards = []
for line in lines:
    match = re.search(r'test_reward: ([\d\.\-]+)', line)
    if match:
        rewards.append(float(match.group(1)))

# 绘制收敛线性图
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Test Reward')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('DRL Convergence Plot')
plt.legend()
plt.grid(True)
plt.show() 