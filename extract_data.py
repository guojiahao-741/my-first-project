import numpy as np

# 定义文件路径
file_paths = [
    '/home/minrui/ISAC/logR3.txt',
    '/home/minrui/ISAC/logR4.txt',
    '/home/minrui/ISAC/logR5.txt',
    '/home/minrui/ISAC/logR6.txt',
    '/home/minrui/ISAC/logR7.txt'
]


# 初始化空列表来存储数据
PPO_rewards = []
PPO_success_rates = []
PPO_pot_rewards = []
PPO_sensing_power = []
PPO_llm_energy = []
PPO_transmission_power = []
PPO_llm_accuracy = []
PPO_crb = []

# 读取文件
for file_path in file_paths:
    rewards = []
    success_rates = []
    pot_rewards = []
    sensing_power = []
    llm_energy = []
    transmission_power = []
    llm_accuracy = []
    crb = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Reward="):
                rewards = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("Success Rate="):
                success_rates = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("PoT Rewards per Node="):
                pot_rewards = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("Sensing Power="):
                sensing_power = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("LLM Energy="):
                llm_energy = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("Transmission Power="):
                transmission_power = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("LLM Accuracy="):
                llm_accuracy = np.array(eval(line.split('=')[1].strip()))
            elif line.startswith("CRB="):
                crb = np.array(eval(line.split('=')[1].strip()))
    
    max_reward_index = np.argmax(np.array(rewards))
    print("The index of the maximum value in the Reward array is:", max_reward_index)
    PPO_rewards.append(rewards[max_reward_index])
    PPO_success_rates.append(success_rates[max_reward_index])
    PPO_pot_rewards.append(pot_rewards[max_reward_index])
    PPO_sensing_power.append(sensing_power[max_reward_index])
    PPO_llm_energy.append(llm_energy[max_reward_index])
    PPO_transmission_power.append(transmission_power[max_reward_index])
    PPO_llm_accuracy.append(llm_accuracy[max_reward_index])
    PPO_crb.append(crb[max_reward_index])
# 打印数组以验证
metrics = ['reward', 'success_rate', 'pot_rewards', 'p_sens', 'e_llm', 'p_tx', 'llm_accuracy', 'crb']
    
print(f"PPO_{metrics[0]}_miner_rewards = ", PPO_rewards)
print(f"PPO_{metrics[1]}_miner_rewards = ", PPO_success_rates)
print(f"PPO_{metrics[2]}_miner_rewards = ", PPO_pot_rewards)
print(f"PPO_{metrics[3]}_miner_rewards = ", PPO_sensing_power)
print(f"PPO_{metrics[4]}_miner_rewards = ", PPO_llm_energy)
print(f"PPO_{metrics[5]}_miner_rewards = ", PPO_transmission_power)
print(f"PPO_{metrics[6]}_miner_rewards = ", PPO_llm_accuracy)
print(f"PPO_{metrics[7]}_miner_rewards = ", PPO_crb)