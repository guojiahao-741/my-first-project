import gymnasium as gym
from gymnasium import spaces

import numpy as np
import os

def compute_crb(node_positions,  # shape (N,2)
                target_positions, # shape (U,2)
                p_rad,           # shape (N,)
                rcs,             # shape (N, N, U)
                H, B, sigma_w, c):
    """
    Compute the CRB for each target using the distributed MIMO radar model.

    Args:
        node_positions: np.ndarray, shape (N,2). node_positions[i] = (x_i, y_i)
        target_positions: np.ndarray, shape (U,2). target_positions[u] = (u_u, v_u)
        p_rad: np.ndarray, shape (N,). p_rad[i] = radar power for node i
        rcs: np.ndarray, shape (N, N, U). rcs[i, j, u] = radar cross-section ell_{i,u,j}
        H: float, common altitude
        B: float, radar bandwidth
        sigma_w: float, noise variance
        c: float, speed of light

    Returns:
        crb_array: np.ndarray of shape (U,), where crb_array[u] = CRB for target u.
    """
    N = node_positions.shape[0]
    U = target_positions.shape[0]

    # Scalar constant xi
    xi = 8.0 * np.pi**2 * (B**2) / (sigma_w * sigma_w * c * c)

    crb_array = np.zeros(U, dtype=np.float64)

    for u in range(U):
        u_u, v_u = target_positions[u]

        # Build b_a, b_b, b_c (each is length N)
        b_a = np.zeros(N, dtype=np.float64)
        b_b = np.zeros(N, dtype=np.float64)
        b_c = np.zeros(N, dtype=np.float64)

        for i in range(N):
            x_i, y_i = node_positions[i]
            R_i_u = np.sqrt((x_i - u_u)**2 + (y_i - v_u)**2 + H**2)

            sum_a = 0.0
            sum_b = 0.0
            sum_c = 0.0

            for j in range(N):
                x_j, y_j = node_positions[j]
                R_j_u = np.sqrt((x_j - u_u)**2 + (y_j - v_u)**2 + H**2)

                alpha_i_u_j = 1.0 / (R_i_u**2 * R_j_u**2)  # path-loss factor
                ell_i_u_j = rcs[i, j, u]

                # bracket terms
                term_x = ((x_i - u_u)/R_i_u + (x_j - u_u)/R_j_u)
                term_y = ((y_i - v_u)/R_i_u + (y_j - v_u)/R_j_u)

                sum_a += alpha_i_u_j * (ell_i_u_j**2) * (term_x**2)
                sum_b += alpha_i_u_j * (ell_i_u_j**2) * (term_y**2)
                sum_c += alpha_i_u_j * (ell_i_u_j**2) * (term_x * term_y)

            b_a[i] = xi * sum_a
            b_b[i] = xi * sum_b
            b_c[i] = xi * sum_c

        # a_u = b_a + b_b
        a_u = b_a + b_b

        # Q_u = b_a b_b^T - b_c b_c^T (outer products)
        Q_u = np.outer(b_a, b_b) - np.outer(b_c, b_c)

        # Convert p_rad to column vector
        p_rad_col = p_rad.reshape((N,1))
        a_u_col = a_u.reshape((N,1))

        numerator = float((a_u_col.T @ p_rad_col) * (p_rad_col.T @ a_u_col))
        denominator = float(p_rad_col.T @ (Q_u @ p_rad_col))

        if denominator < 1e-12:
            crb_value = 1e12
        else:
            crb_value = numerator / denominator

        crb_array[u] = crb_value

    return crb_array


class BlockchainISACEnvWithRequirements(gym.Env):
    """
    An environment where each target has a required CRB.
    If the current CRB does not meet these requirements, the reward is 0.
    We also record the success rate over time.
    """

    def __init__(self,
                 N=3,            # number of nodes
                 U=5,            # number of targets
                 max_episode_steps=100,
                 p_max=1.0,
                 H=100.0,        # altitude
                 B=50e6,         # radar bandwidth
                 sigma_w=1e-11,  # noise variance
                 c=3e8,
                 max_compute_units=12,  # maximum number of compute units per node
                 miner_percentage=0.3,  # percentage of miner nodes
                 miner_reward=0.5,      # fixed reward for miners
                 seed=42):
        super().__init__()
        self.N = N
        self.U = U
        self.max_episode_steps = max_episode_steps
        self.p_max = p_max
        self.H = H
        self.B = B
        self.sigma_w = sigma_w
        self.c = c
        self.rng = np.random.default_rng(seed)
        
        # PoT reward parameters
        self.miner_reward = miner_reward  # fixed reward for miners
        
        # Calculate number of miners and workers
        self.miner_count = max(1, int(N * miner_percentage))
        self.worker_count = N - self.miner_count
        
        # Initialize node roles (0: worker node, 1: miner node)
        self.node_roles = np.zeros(N, dtype=np.int32)
        self.node_roles[:self.miner_count] = 1  # first miner_count nodes are miners
        
        # Initialize node stakes
        self.node_stakes = np.ones(N, dtype=np.float32)
        
        # Randomly assign compute power (number of compute units) to each node
        self.compute_units = self.rng.integers(1, max_compute_units + 1, size=N, dtype=np.int32)
        # Energy efficiency per compute unit (energy/compute unit)
        self.compute_efficiency = np.array([0.1, 0.15, 0.2, 0.12, 0.18], dtype=np.float32)[:N]
        # Maximum compute capacity per compute unit (FLOPS)
        self.compute_capacity = np.array([1e9, 2e9, 3e9, 1.5e9, 2.5e9], dtype=np.float32)[:N]

        # Add LLM accuracy related parameters
        self.base_accuracy = 0.8  # base accuracy
        self.compute_threshold = 1e9  # compute power threshold (FLOPS)
        self.accuracy_sensitivity = 0.2  # accuracy sensitivity to compute power

        # Generate different LLM accuracy requirements for each target at each timestep
        # Base accuracy requirement range: 0.7-0.9
        self.required_accuracy = self.rng.uniform(0.3, 1, size=(max_episode_steps, U)).astype(np.float32)

        # Example geometry (adapt as needed)
        # Position of up to 5 ISAC nodes forming a coverage area
        self.node_positions = np.array([
            [0,     0],      # bottom-left node
            [200,   0],      # bottom-right node
            [100,   173],    # top-center node
            [50,    50],     # middle-left node
            [150,   50]      # middle-right node
        ], dtype=np.float32)[:N]

        # Position of up to 10 targets uniformly distributed in the coverage area
        self.target_positions = np.array([
            [50,    30],     # bottom-left area
            [150,   30],     # bottom-right area
            [100,   100],    # center area
            [30,    80],     # top-left area
            [170,   80],     # top-right area
            [70,    120],    # top-middle area
            [130,   120],    # top-middle area
            [20,    50],     # left area
            [180,   50],     # right area
            [100,   150]     # top area
        ], dtype=np.float32)[:U]

        # Radar cross-section: shape (N,N,U)
        self.rcs = np.ones((N, N, U), dtype=np.float32)

        # Generate different CRB requirements for each target at each timestep
        # Base CRB requirement range: 50-150
        self.required_crb = self.rng.uniform(6, 12, size=(max_episode_steps, U)).astype(np.float32)

        # Calculate total observation space size
        obs_size = U + N + N + N + U + U + N + N  # CRB + energy + compute + llm + req_crb + req_acc + roles + stakes

        # Define observation space as a Box
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Each node picks (p_sens, e_llm, p_tx)
        self.action_space = spaces.Box(
            low=0.0, high=p_max, shape=(3 * N,), dtype=np.float32
        )

        # Initialize initial state
        self.crb_init = self.rng.uniform(5, 10, size=(U,)).astype(np.float32)
        self.energy_init = np.full((N,), fill_value=p_max*max_episode_steps, dtype=np.float32)
        self.compute_status_init = np.zeros((N,), dtype=np.float32)  # initial compute status
        self.llm_accuracy_init = np.zeros((N,), dtype=np.float32)  # initial LLM accuracy

        # Initialize statistics
        self.success_count = 0
        self.total_count = 0

        # Initialize current state
        self._state_dict = {
            "CRB": self.crb_init.copy(),
            "energy_levels": self.energy_init.copy(),
            "compute_status": self.compute_status_init.copy(),
            "llm_accuracy": self.llm_accuracy_init.copy(),
            "required_crb": self.required_crb[0].copy(),  # initial CRB requirement
            "required_accuracy": self.required_accuracy[0].copy(),  # initial accuracy requirement
            "node_roles": self.node_roles.copy(),  # node roles
            "node_stakes": self.node_stakes.copy()  # node stakes
        }
        self.state = self._dict_to_array(self._state_dict)
        
        # environment records
        # reward records
        self.all_reward_records = []
        # success records
        self.all_success_records = []
        # 添加新的记录列表
        self.all_pot_rewards_records = []
        self.all_p_sens_records = []
        self.all_e_llm_records = []
        self.all_p_tx_records = []
        self.all_llm_accuracy_records = []
        self.all_new_crb_records = []
        
        # 添加当前episode的记录列表
        self.episode_pot_rewards = []
        self.episode_p_sens = []
        self.episode_e_llm = []
        self.episode_p_tx = []
        self.episode_llm_accuracy = []
        self.episode_new_crb = []

    def _dict_to_array(self, state_dict):
        """Convert state dictionary to flat array."""
        return np.concatenate([
            state_dict["CRB"],
            state_dict["energy_levels"],
            state_dict["compute_status"],
            state_dict["llm_accuracy"],
            state_dict["required_crb"],
            state_dict["required_accuracy"],
            state_dict["node_roles"],
            state_dict["node_stakes"]
        ])

    def compute_llm_accuracy(self, effective_compute):
        """
        Calculate LLM accuracy based on actual compute power
        Args:
            effective_compute: actual compute power (FLOPS)
        Returns:
            accuracy: LLM accuracy
        """
        # Compute power to threshold ratio
        compute_ratio = effective_compute / self.compute_threshold
        # Use sigmoid function to calculate accuracy, ensure in [0,1] range
        # When compute power is 0, accuracy should be close to base_accuracy - sensitivity
        # When compute power equals threshold, accuracy should be close to base_accuracy
        # When compute power exceeds threshold, accuracy should be close to base_accuracy + sensitivity
        accuracy = self.base_accuracy + self.accuracy_sensitivity * (1 / (1 + np.exp(-compute_ratio)) - 0.5)
        # Ensure accuracy is within reasonable range
        return np.clip(accuracy, self.base_accuracy - self.accuracy_sensitivity, self.base_accuracy + self.accuracy_sensitivity)

    def compute_pot_rewards(self, llm_accuracy, success):
        """
        Calculate PoT rewards based on node roles and performance
        
        Args:
            llm_accuracy: LLM accuracy for each node
            success: whether all requirements are met
            
        Returns:
            pot_rewards: PoT rewards for each node
        """
        pot_rewards = np.zeros(self.N, dtype=np.float32)
        
        # If task is successful, calculate rewards according to PoT model
        if success:
            # Select main miner (miner node with highest stake)
            miner_indices = np.where(self.node_roles == 1)[0]
            if len(miner_indices) > 0:
                main_miner_idx = miner_indices[np.argmax(self.node_stakes[miner_indices])]
                
                # Assign accuracy-based rewards to worker nodes
                for i in range(self.N):
                    if self.node_roles[i] == 0:  # worker node
                        # Use LLM accuracy as semantic evaluation score
                        pot_rewards[i] = llm_accuracy[i]
                    elif self.node_roles[i] == 1:  # miner node
                        if i == main_miner_idx:  # main miner
                            pot_rewards[i] = self.miner_reward
                        else:  # other miners
                            pot_rewards[i] = 0.0
        
        # Update node stakes
        self.node_stakes += pot_rewards
        
        # Periodically reallocate roles based on stakes (every 10 steps)
        if self.current_step % 10 == 0:
            self.reallocate_roles()
            
        return pot_rewards
    
    def reallocate_roles(self):
        """
        Reallocate roles based on node stakes
        """
        # Get nodes with highest stakes as miners
        top_indices = np.argsort(-self.node_stakes)[:self.miner_count]
        
        # Reset all node roles
        self.node_roles.fill(0)
        
        # Set nodes with highest stakes as miners
        for idx in top_indices:
            self.node_roles[idx] = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset current step
        self.current_step = 0
        
        # Reset state dictionary
        self._state_dict = {
            "CRB": self.crb_init.copy(),
            "energy_levels": self.energy_init.copy(),
            "compute_status": self.compute_status_init.copy(),
            "llm_accuracy": self.llm_accuracy_init.copy(),
            "required_crb": self.required_crb[0].copy(),
            "required_accuracy": self.required_accuracy[0].copy(),
            "node_roles": self.node_roles.copy(),
            "node_stakes": self.node_stakes.copy()
        }
        
        # Convert to flat array
        self.state = self._dict_to_array(self._state_dict)

        self.reward_records = []
        self.success_records = []
        
        # 重置当前episode的记录列表
        self.episode_pot_rewards = []
        self.episode_p_sens = []
        self.episode_e_llm = []
        self.episode_p_tx = []
        self.episode_llm_accuracy = []
        self.episode_new_crb = []
        
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.total_count += 1  # we always count each step as a "trial"

        # Reshape action from (3*N,) to (N,3)
        action = action.reshape(-1, 3)
        # print("action:", action)
        
        # Decompose action
        p_sens = action[:, 0]  # sensing power
        e_llm = action[:, 1]   # LLM energy
        p_tx = action[:, 2]    # transmission power

        # Enforce p_max constraint
        for i in range(self.N):
            total_power = p_sens[i] + e_llm[i] + p_tx[i]
            if total_power > self.p_max:
                factor = self.p_max / (total_power + 1e-8)
                p_sens[i] *= factor
                e_llm[i] *= factor
                p_tx[i] *= factor

        # Calculate actual compute power for each node
        effective_compute = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            # Calculate actual available compute units (based on energy)
            available_units = min(
                self.compute_units[i],
                int(e_llm[i] / self.compute_efficiency[i])
            )
            # Calculate actual compute power
            effective_compute[i] = available_units * self.compute_capacity[i]

        # Calculate LLM accuracy
        llm_accuracy = self.compute_llm_accuracy(effective_compute)
        # print("llm_accuracy:", llm_accuracy)

        # Cost = sum of each node's action
        costs_per_node = p_sens + e_llm + p_tx
        
        # Evaluate task success before calculating PoT rewards
        # CRB update with p_rad = p_sens
        new_crb = compute_crb(
            node_positions=self.node_positions,
            target_positions=self.target_positions,
            p_rad=p_sens,
            rcs=self.rcs,
            H=self.H, B=self.B, sigma_w=self.sigma_w, c=self.c
        ).astype(np.float32)
        
        # Get current CRB and accuracy requirements
        current_required_crb = self.required_crb[self.current_step - 1]
        current_required_accuracy = self.required_accuracy[self.current_step - 1]

        # Calculate node accuracy for each target (assume one node per target)
        target_accuracy = np.zeros(self.U, dtype=np.float32)
        for u in range(self.U):
            # Select node responsible for this target (simply choose first node here)
            target_accuracy[u] = llm_accuracy[0]

        # Check if *all* targets satisfy both CRB and accuracy requirements
        crb_ok = np.all(new_crb <= current_required_crb)
        accuracy_ok = np.all(target_accuracy >= current_required_accuracy)
        success = crb_ok and accuracy_ok
        
        # Calculate PoT rewards based on node roles and performance
        reward_scale = 5
        pot_rewards_per_node = reward_scale * self.compute_pot_rewards(llm_accuracy, success)
        
        # net node rewards
        node_rewards = pot_rewards_per_node - costs_per_node
        reward = float(np.sum(node_rewards))  # the "would-be" reward

        # Update energy
        new_energy = self.state[self.U:self.U+self.N] - costs_per_node
        new_energy = np.clip(new_energy, 0.0, None)

        # Update compute status (normalized compute power)
        new_compute_status = effective_compute / (self.compute_units * self.compute_capacity)
        # reward_scale = 10
        # if success:
        #     # Requirements met => keep the raw reward
        #     reward = raw_reward
        # else:
        #     # Requirements not met => zero reward
        #     reward = raw_reward

        # Update success statistics
        if success:
            self.success_count += 1
        success_rate = self.success_count / self.current_step

        self.reward_records.append(reward)
        self.success_records.append(success)

        # Update state dictionary
        self._state_dict = {
            "CRB": new_crb,
            "energy_levels": new_energy,
            "compute_status": new_compute_status,
            "llm_accuracy": llm_accuracy,
            "required_crb": current_required_crb,
            "required_accuracy": current_required_accuracy,
            "node_roles": self.node_roles.copy(),
            "node_stakes": self.node_stakes.copy()
        }
        
        # Convert to flat array
        self.state = self._dict_to_array(self._state_dict)

        # 在每个step中记录当前值
        self.episode_pot_rewards.append(np.mean(pot_rewards_per_node))
        self.episode_p_sens.append(np.mean(p_sens))
        self.episode_e_llm.append(np.mean(e_llm))
        self.episode_p_tx.append(np.mean(p_tx))
        self.episode_llm_accuracy.append(np.mean(llm_accuracy))
        self.episode_new_crb.append(np.mean(new_crb))

        # Check termination
        done = False
        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True
            self.all_reward_records.append(np.mean(self.reward_records))
            self.all_success_records.append(np.sum(self.success_records)/self.max_episode_steps)
            # 记录当前episode的平均值
            self.all_pot_rewards_records.append(np.mean(self.episode_pot_rewards))
            self.all_p_sens_records.append(np.mean(self.episode_p_sens))
            self.all_e_llm_records.append(np.mean(self.episode_e_llm))
            self.all_p_tx_records.append(np.mean(self.episode_p_tx))
            self.all_llm_accuracy_records.append(np.mean(self.episode_llm_accuracy))
            self.all_new_crb_records.append(np.mean(self.episode_new_crb))
        if np.all(new_energy <= 1e-5):
            done = True
            self.all_reward_records.append(np.mean(self.reward_records))
            self.all_success_records.append(np.sum(self.success_records)/self.max_episode_steps)
            # 记录当前episode的平均值
            self.all_pot_rewards_records.append(np.mean(self.episode_pot_rewards))
            self.all_p_sens_records.append(np.mean(self.episode_p_sens))
            self.all_e_llm_records.append(np.mean(self.episode_e_llm))
            self.all_p_tx_records.append(np.mean(self.episode_p_tx))
            self.all_llm_accuracy_records.append(np.mean(self.episode_llm_accuracy))
            self.all_new_crb_records.append(np.mean(self.episode_new_crb))

        info = {
            "node_rewards": node_rewards,
            "pot_rewards": pot_rewards_per_node,
            "costs": costs_per_node,
            "p_sens": p_sens,
            "e_llm": e_llm,
            "p_tx": p_tx,
            "effective_compute": effective_compute,
            "llm_accuracy": llm_accuracy,
            "target_accuracy": target_accuracy,
            "success_rate": success_rate,
            "success": success,
            "required_crb": current_required_crb,
            "required_accuracy": current_required_accuracy,
            "node_roles": self.node_roles,
            "node_stakes": self.node_stakes
        }
        return self.state, reward, done, truncated, info

    def render(self):
        """打印当前环境状态"""
        print("\n=====Environment State=====")
        print(f"Reward= {self.all_reward_records}")
        print(f"Success Rate= {self.all_success_records}")
        print("\n=====Detailed Metrics=====")
        print("PoT Rewards per Node=", self.all_pot_rewards_records)
        print("Sensing Power=", self.all_p_sens_records)
        print("LLM Energy=", self.all_e_llm_records)
        print("Transmission Power=", self.all_p_tx_records)
        print("LLM Accuracy=", self.all_llm_accuracy_records)
        print("CRB=", self.all_new_crb_records)

    def close(self):
        pass

    @staticmethod
    def greedy_algorithm(env, obs):
        """
        Greedy algorithm for selecting optimal actions in ISAC environment
        
        Strategy:
        1. Allocate sufficient LLM energy to meet accuracy requirements
        2. Allocate sufficient sensing power to meet CRB requirements
        3. Allocate remaining energy to transmission power
        """
        N = env.N
        U = env.U
        p_max = env.p_max
        
        # Get current requirements from state dictionary
        required_crb = env._state_dict["required_crb"]
        required_accuracy = env._state_dict["required_accuracy"]
        
        # Initialize action
        action = np.zeros((N, 3), dtype=np.float32)
        
        # Step 1: Calculate LLM energy needed to meet accuracy requirements
        for i in range(N):
            # Find highest accuracy requirement for current node
            max_required_accuracy = np.max(required_accuracy)
            
            # If accuracy requirement is above base_accuracy + sensitivity, set to maximum
            if max_required_accuracy > env.base_accuracy + env.accuracy_sensitivity:
                max_required_accuracy = env.base_accuracy + env.accuracy_sensitivity
            
            # Reverse calculate required compute power for this accuracy
            if max_required_accuracy <= env.base_accuracy - env.accuracy_sensitivity:
                required_compute = 0
            else:
                try:
                    # Calculate required compute ratio from accuracy
                    accuracy_diff = max_required_accuracy - env.base_accuracy
                    sigmoid_value = accuracy_diff / env.accuracy_sensitivity + 0.5
                    
                    # Handle boundary cases safely
                    if sigmoid_value >= 0.99:
                        compute_ratio = 10  # a sufficiently large value
                    elif sigmoid_value <= 0.01:
                        compute_ratio = -10  # a sufficiently small value
                    else:
                        # Avoid log(negative) or log(0)
                        denominator = sigmoid_value
                        if denominator >= 1.0:
                            denominator = 0.99
                        elif denominator <= 0.0:
                            denominator = 0.01
                        
                        compute_ratio = np.log(denominator / (1 - denominator))
                    
                    required_compute = compute_ratio * env.compute_threshold
                    # Ensure compute power is positive
                    required_compute = max(0, required_compute)
                except Exception as e:
                    print(f"Calculation error: {e}, using default compute power")
                    # Use default value on error
                    required_compute = env.compute_threshold
            
            # Calculate required compute units
            try:
                required_units = min(
                    env.compute_units[i],
                    max(1, int(np.ceil(required_compute / env.compute_capacity[i])))
                )
            except:
                # If error occurs, use half of compute units
                required_units = max(1, env.compute_units[i] // 2)
            
            # Calculate required LLM energy with some margin for better accuracy
            action[i, 1] = required_units * env.compute_efficiency[i] * 1.2  # add 20% energy margin
            # Ensure not exceeding p_max
            action[i, 1] = min(action[i, 1], p_max)
        
        # Step 2: Calculate sensing power needed to meet CRB requirements (binary search)
        # Use binary search to find appropriate sensing power faster
        left, right = 0.05, p_max
        best_p_sens = None
        best_crb = None
        
        while right - left > 0.01:
            mid = (left + right) / 2
            temp_action = action.copy()
            temp_action[:, 0] = mid
            
            # Check if this sensing power meets CRB requirements
            temp_p_sens = temp_action[:, 0]
            temp_crb = compute_crb(
                node_positions=env.node_positions,
                target_positions=env.target_positions,
                p_rad=temp_p_sens,
                rcs=env.rcs,
                H=env.H, B=env.B, sigma_w=env.sigma_w, c=env.c
            ).astype(np.float32)
            
            if np.all(temp_crb <= required_crb):
                best_p_sens = mid
                best_crb = temp_crb
                right = mid  # try smaller power
            else:
                left = mid  # need larger power
        
        # Use found best sensing power
        if best_p_sens is not None:
            action[:, 0] = best_p_sens
        else:
            # If no suitable power found, use maximum power
            action[:, 0] = np.minimum(p_max * 0.8, p_max - action[:, 1])  # use 80% of max power
        
        # Step 3: Optimize transmission power allocation
        for i in range(N):
            # Calculate remaining available power
            remaining_power = p_max - (action[i, 0] + action[i, 1])
            
            # Allocate transmission power based on node role
            if env.node_roles[i] == 1:  # miner node
                # Miner nodes get more transmission power
                action[i, 2] = remaining_power * 0.8  # use 80% of remaining power
            else:  # worker node
                # Worker nodes get less transmission power
                action[i, 2] = remaining_power * 0.5  # use 50% of remaining power
        
        return action.flatten()  # 返回展平的动作数组

def test_different_N():
    """
    测试不同N值(1-5)下random和greedy策略的表现
    """
    # 确保保存目录存在
    save_dir = "/home/minrui/ISAC"
    os.makedirs(save_dir, exist_ok=True)
    
    N_values = range(2, 6)
    results = []
    
    # 创建指标列表
    N_list = []
    random_reward_list = []
    random_success_rate_list = []
    random_pot_rewards_list = []
    random_p_sens_list = []
    random_e_llm_list = []
    random_p_tx_list = []
    random_llm_accuracy_list = []
    random_crb_list = []
    
    greedy_reward_list = []
    greedy_success_rate_list = []
    greedy_pot_rewards_list = []
    greedy_p_sens_list = []
    greedy_e_llm_list = []
    greedy_p_tx_list = []
    greedy_llm_accuracy_list = []
    greedy_crb_list = []
    
    for N in N_values:
        print(f"\n===== 测试节点数量 N = {N} =====")
        
        # 创建环境
        random_env = BlockchainISACEnvWithRequirements(N=N, U=5, max_episode_steps=100, p_max=1.0, seed=42)
        greedy_env = BlockchainISACEnvWithRequirements(N=N, U=5, max_episode_steps=100, p_max=1.0, seed=42)
        
        # 测试random策略
        print("\n测试random策略:")
        obs, info = random_env.reset()
        for i in range(100):
            action = random_env.action_space.sample()
            obs, reward, done, truncated, info = random_env.step(action)
            if done or truncated:
                obs, info = random_env.reset()
        
        # 测试greedy策略
        print("\n测试greedy策略:")
        obs, info = greedy_env.reset()
        for i in range(100):
            action = BlockchainISACEnvWithRequirements.greedy_algorithm(greedy_env, obs)
            obs, reward, done, truncated, info = greedy_env.step(action)
            if done or truncated:
                obs, info = greedy_env.reset()
        
        # 记录结果
        result = {
            'N': N,
            'random_reward': np.mean(random_env.all_reward_records),
            'random_success_rate': np.mean(random_env.all_success_records),
            'greedy_reward': np.mean(greedy_env.all_reward_records),
            'greedy_success_rate': np.mean(greedy_env.all_success_records),
            'random_pot_rewards': np.mean(random_env.all_pot_rewards_records),
            'random_p_sens': np.mean(random_env.all_p_sens_records),
            'random_e_llm': np.mean(random_env.all_e_llm_records),
            'random_p_tx': np.mean(random_env.all_p_tx_records),
            'random_llm_accuracy': np.mean(random_env.all_llm_accuracy_records),
            'random_crb': np.mean(random_env.all_new_crb_records),
            'greedy_pot_rewards': np.mean(greedy_env.all_pot_rewards_records),
            'greedy_p_sens': np.mean(greedy_env.all_p_sens_records),
            'greedy_e_llm': np.mean(greedy_env.all_e_llm_records),
            'greedy_p_tx': np.mean(greedy_env.all_p_tx_records),
            'greedy_llm_accuracy': np.mean(greedy_env.all_llm_accuracy_records),
            'greedy_crb': np.mean(greedy_env.all_new_crb_records)
        }
        results.append(result)
        
        # 添加到指标列表
        N_list.append(N)
        random_reward_list.append(result['random_reward'])
        random_success_rate_list.append(result['random_success_rate'])
        random_pot_rewards_list.append(result['random_pot_rewards'])
        random_p_sens_list.append(result['random_p_sens'])
        random_e_llm_list.append(result['random_e_llm'])
        random_p_tx_list.append(result['random_p_tx'])
        random_llm_accuracy_list.append(result['random_llm_accuracy'])
        random_crb_list.append(result['random_crb'])
        
        greedy_reward_list.append(result['greedy_reward'])
        greedy_success_rate_list.append(result['greedy_success_rate'])
        greedy_pot_rewards_list.append(result['greedy_pot_rewards'])
        greedy_p_sens_list.append(result['greedy_p_sens'])
        greedy_e_llm_list.append(result['greedy_e_llm'])
        greedy_p_tx_list.append(result['greedy_p_tx'])
        greedy_llm_accuracy_list.append(result['greedy_llm_accuracy'])
        greedy_crb_list.append(result['greedy_crb'])
        
        random_env.close()
        greedy_env.close()
    
    # 保存数据到文件
    data = {
        'N': N_list,
        'random_reward': random_reward_list,
        'random_success_rate': random_success_rate_list,
        'greedy_reward': greedy_reward_list,
        'greedy_success_rate': greedy_success_rate_list,
        'random_pot_rewards': random_pot_rewards_list,
        'random_p_sens': random_p_sens_list,
        'random_e_llm': random_e_llm_list,
        'random_p_tx': random_p_tx_list,
        'random_llm_accuracy': random_llm_accuracy_list,
        'random_crb': random_crb_list,
        'greedy_pot_rewards': greedy_pot_rewards_list,
        'greedy_p_sens': greedy_p_sens_list,
        'greedy_e_llm': greedy_e_llm_list,
        'greedy_p_tx': greedy_p_tx_list,
        'greedy_llm_accuracy': greedy_llm_accuracy_list,
        'greedy_crb': greedy_crb_list
    }
    np.save(os.path.join(save_dir, 'test_results_N.npy'), data)
    
    # 打印总结
    print("\n===== 总结 =====")
    print("\nRandom策略:")
    print(f"N值: {N_list}")
    print(f"平均奖励: {random_reward_list}")
    print(f"成功率: {random_success_rate_list}")
    print(f"PoT奖励: {random_pot_rewards_list}")
    print(f"感知功率: {random_p_sens_list}")
    print(f"LLM能量: {random_e_llm_list}")
    print(f"传输功率: {random_p_tx_list}")
    print(f"LLM准确率: {random_llm_accuracy_list}")
    print(f"CRB: {random_crb_list}")
    
    print("\nGreedy策略:")
    print(f"N值: {N_list}")
    print(f"平均奖励: {greedy_reward_list}")
    print(f"成功率: {greedy_success_rate_list}")
    print(f"PoT奖励: {greedy_pot_rewards_list}")
    print(f"感知功率: {greedy_p_sens_list}")
    print(f"LLM能量: {greedy_e_llm_list}")
    print(f"传输功率: {greedy_p_tx_list}")
    print(f"LLM准确率: {greedy_llm_accuracy_list}")
    print(f"CRB: {greedy_crb_list}")

def test_different_U():
    """
    测试不同U值(4-8)下random和greedy策略的表现
    """
    # 确保保存目录存在
    save_dir = "/home/minrui/ISAC"
    os.makedirs(save_dir, exist_ok=True)
    
    U_values = range(4, 9)
    results = []
    
    # 创建指标列表
    U_list = []
    random_reward_list = []
    random_success_rate_list = []
    random_pot_rewards_list = []
    random_p_sens_list = []
    random_e_llm_list = []
    random_p_tx_list = []
    random_llm_accuracy_list = []
    random_crb_list = []
    
    greedy_reward_list = []
    greedy_success_rate_list = []
    greedy_pot_rewards_list = []
    greedy_p_sens_list = []
    greedy_e_llm_list = []
    greedy_p_tx_list = []
    greedy_llm_accuracy_list = []
    greedy_crb_list = []
    
    for U in U_values:
        print(f"\n===== 测试目标数量 U = {U} =====")
        
        # 创建环境
        random_env = BlockchainISACEnvWithRequirements(N=3, U=U, max_episode_steps=100, p_max=1.0, seed=42)
        greedy_env = BlockchainISACEnvWithRequirements(N=3, U=U, max_episode_steps=100, p_max=1.0, seed=42)
        
        # 测试random策略
        print("\n测试random策略:")
        obs, info = random_env.reset()
        for i in range(100):
            action = random_env.action_space.sample()
            obs, reward, done, truncated, info = random_env.step(action)
            if done or truncated:
                obs, info = random_env.reset()
        
        # 测试greedy策略
        print("\n测试greedy策略:")
        obs, info = greedy_env.reset()
        for i in range(100):
            action = BlockchainISACEnvWithRequirements.greedy_algorithm(greedy_env, obs)
            obs, reward, done, truncated, info = greedy_env.step(action)
            if done or truncated:
                obs, info = greedy_env.reset()
        
        # 记录结果
        result = {
            'U': U,
            'random_reward': np.mean(random_env.all_reward_records),
            'random_success_rate': np.mean(random_env.all_success_records),
            'greedy_reward': np.mean(greedy_env.all_reward_records),
            'greedy_success_rate': np.mean(greedy_env.all_success_records),
            'random_pot_rewards': np.mean(random_env.all_pot_rewards_records),
            'random_p_sens': np.mean(random_env.all_p_sens_records),
            'random_e_llm': np.mean(random_env.all_e_llm_records),
            'random_p_tx': np.mean(random_env.all_p_tx_records),
            'random_llm_accuracy': np.mean(random_env.all_llm_accuracy_records),
            'random_crb': np.mean(random_env.all_new_crb_records),
            'greedy_pot_rewards': np.mean(greedy_env.all_pot_rewards_records),
            'greedy_p_sens': np.mean(greedy_env.all_p_sens_records),
            'greedy_e_llm': np.mean(greedy_env.all_e_llm_records),
            'greedy_p_tx': np.mean(greedy_env.all_p_tx_records),
            'greedy_llm_accuracy': np.mean(greedy_env.all_llm_accuracy_records),
            'greedy_crb': np.mean(greedy_env.all_new_crb_records)
        }
        results.append(result)
        
        # 添加到指标列表
        U_list.append(U)
        random_reward_list.append(result['random_reward'])
        random_success_rate_list.append(result['random_success_rate'])
        random_pot_rewards_list.append(result['random_pot_rewards'])
        random_p_sens_list.append(result['random_p_sens'])
        random_e_llm_list.append(result['random_e_llm'])
        random_p_tx_list.append(result['random_p_tx'])
        random_llm_accuracy_list.append(result['random_llm_accuracy'])
        random_crb_list.append(result['random_crb'])
        
        greedy_reward_list.append(result['greedy_reward'])
        greedy_success_rate_list.append(result['greedy_success_rate'])
        greedy_pot_rewards_list.append(result['greedy_pot_rewards'])
        greedy_p_sens_list.append(result['greedy_p_sens'])
        greedy_e_llm_list.append(result['greedy_e_llm'])
        greedy_p_tx_list.append(result['greedy_p_tx'])
        greedy_llm_accuracy_list.append(result['greedy_llm_accuracy'])
        greedy_crb_list.append(result['greedy_crb'])
        
        random_env.close()
        greedy_env.close()
    
    # 保存数据到文件
    data = {
        'U': U_list,
        'random_reward': random_reward_list,
        'random_success_rate': random_success_rate_list,
        'greedy_reward': greedy_reward_list,
        'greedy_success_rate': greedy_success_rate_list,
        'random_pot_rewards': random_pot_rewards_list,
        'random_p_sens': random_p_sens_list,
        'random_e_llm': random_e_llm_list,
        'random_p_tx': random_p_tx_list,
        'random_llm_accuracy': random_llm_accuracy_list,
        'random_crb': random_crb_list,
        'greedy_pot_rewards': greedy_pot_rewards_list,
        'greedy_p_sens': greedy_p_sens_list,
        'greedy_e_llm': greedy_e_llm_list,
        'greedy_p_tx': greedy_p_tx_list,
        'greedy_llm_accuracy': greedy_llm_accuracy_list,
        'greedy_crb': greedy_crb_list
    }
    np.save(os.path.join(save_dir, 'test_results_U.npy'), data)
    
    # 打印总结
    print("\n===== 总结 =====")
    print("\nRandom策略:")
    print(f"U值: {U_list}")
    print(f"平均奖励: {random_reward_list}")
    print(f"成功率: {random_success_rate_list}")
    print(f"PoT奖励: {random_pot_rewards_list}")
    print(f"感知功率: {random_p_sens_list}")
    print(f"LLM能量: {random_e_llm_list}")
    print(f"传输功率: {random_p_tx_list}")
    print(f"LLM准确率: {random_llm_accuracy_list}")
    print(f"CRB: {random_crb_list}")
    
    print("\nGreedy策略:")
    print(f"U值: {U_list}")
    print(f"平均奖励: {greedy_reward_list}")
    print(f"成功率: {greedy_success_rate_list}")
    print(f"PoT奖励: {greedy_pot_rewards_list}")
    print(f"感知功率: {greedy_p_sens_list}")
    print(f"LLM能量: {greedy_e_llm_list}")
    print(f"传输功率: {greedy_p_tx_list}")
    print(f"LLM准确率: {greedy_llm_accuracy_list}")
    print(f"CRB: {greedy_crb_list}")

def test_different_compute_units():
    """
    测试不同max-compute-units值(8-16)下random和greedy策略的表现
    """
    # 确保保存目录存在
    save_dir = "/home/minrui/ISAC"
    os.makedirs(save_dir, exist_ok=True)
    
    compute_units_values = [8, 10, 12, 14, 16]
    results = []
    
    # 创建指标列表
    compute_units_list = []
    random_reward_list = []
    random_success_rate_list = []
    random_pot_rewards_list = []
    random_p_sens_list = []
    random_e_llm_list = []
    random_p_tx_list = []
    random_llm_accuracy_list = []
    random_crb_list = []
    
    greedy_reward_list = []
    greedy_success_rate_list = []
    greedy_pot_rewards_list = []
    greedy_p_sens_list = []
    greedy_e_llm_list = []
    greedy_p_tx_list = []
    greedy_llm_accuracy_list = []
    greedy_crb_list = []
    
    for max_compute_units in compute_units_values:
        print(f"\n===== 测试最大计算单元数 max_compute_units = {max_compute_units} =====")
        
        # 创建环境
        random_env = BlockchainISACEnvWithRequirements(N=3, U=5, max_episode_steps=100, p_max=1.0, max_compute_units=max_compute_units, seed=42)
        greedy_env = BlockchainISACEnvWithRequirements(N=3, U=5, max_episode_steps=100, p_max=1.0, max_compute_units=max_compute_units, seed=42)
        
        # 测试random策略
        print("\n测试random策略:")
        obs, info = random_env.reset()
        for i in range(100):
            action = random_env.action_space.sample()
            obs, reward, done, truncated, info = random_env.step(action)
            if done or truncated:
                obs, info = random_env.reset()
        
        # 测试greedy策略
        print("\n测试greedy策略:")
        obs, info = greedy_env.reset()
        for i in range(100):
            action = BlockchainISACEnvWithRequirements.greedy_algorithm(greedy_env, obs)
            obs, reward, done, truncated, info = greedy_env.step(action)
            if done or truncated:
                obs, info = greedy_env.reset()
        
        # 记录结果
        result = {
            'max_compute_units': max_compute_units,
            'random_reward': np.mean(random_env.all_reward_records),
            'random_success_rate': np.mean(random_env.all_success_records),
            'greedy_reward': np.mean(greedy_env.all_reward_records),
            'greedy_success_rate': np.mean(greedy_env.all_success_records),
            'random_pot_rewards': np.mean(random_env.all_pot_rewards_records),
            'random_p_sens': np.mean(random_env.all_p_sens_records),
            'random_e_llm': np.mean(random_env.all_e_llm_records),
            'random_p_tx': np.mean(random_env.all_p_tx_records),
            'random_llm_accuracy': np.mean(random_env.all_llm_accuracy_records),
            'random_crb': np.mean(random_env.all_new_crb_records),
            'greedy_pot_rewards': np.mean(greedy_env.all_pot_rewards_records),
            'greedy_p_sens': np.mean(greedy_env.all_p_sens_records),
            'greedy_e_llm': np.mean(greedy_env.all_e_llm_records),
            'greedy_p_tx': np.mean(greedy_env.all_p_tx_records),
            'greedy_llm_accuracy': np.mean(greedy_env.all_llm_accuracy_records),
            'greedy_crb': np.mean(greedy_env.all_new_crb_records)
        }
        results.append(result)
        
        # 添加到指标列表
        compute_units_list.append(max_compute_units)
        random_reward_list.append(result['random_reward'])
        random_success_rate_list.append(result['random_success_rate'])
        random_pot_rewards_list.append(result['random_pot_rewards'])
        random_p_sens_list.append(result['random_p_sens'])
        random_e_llm_list.append(result['random_e_llm'])
        random_p_tx_list.append(result['random_p_tx'])
        random_llm_accuracy_list.append(result['random_llm_accuracy'])
        random_crb_list.append(result['random_crb'])
        
        greedy_reward_list.append(result['greedy_reward'])
        greedy_success_rate_list.append(result['greedy_success_rate'])
        greedy_pot_rewards_list.append(result['greedy_pot_rewards'])
        greedy_p_sens_list.append(result['greedy_p_sens'])
        greedy_e_llm_list.append(result['greedy_e_llm'])
        greedy_p_tx_list.append(result['greedy_p_tx'])
        greedy_llm_accuracy_list.append(result['greedy_llm_accuracy'])
        greedy_crb_list.append(result['greedy_crb'])
    
    random_env.close()
    greedy_env.close()

    # 保存数据到文件
    data = {
        'compute_units': compute_units_list,
        'random_reward': random_reward_list,
        'random_success_rate': random_success_rate_list,
        'greedy_reward': greedy_reward_list,
        'greedy_success_rate': greedy_success_rate_list,
        'random_pot_rewards': random_pot_rewards_list,
        'random_p_sens': random_p_sens_list,
        'random_e_llm': random_e_llm_list,
        'random_p_tx': random_p_tx_list,
        'random_llm_accuracy': random_llm_accuracy_list,
        'random_crb': random_crb_list,
        'greedy_pot_rewards': greedy_pot_rewards_list,
        'greedy_p_sens': greedy_p_sens_list,
        'greedy_e_llm': greedy_e_llm_list,
        'greedy_p_tx': greedy_p_tx_list,
        'greedy_llm_accuracy': greedy_llm_accuracy_list,
        'greedy_crb': greedy_crb_list
    }
    np.save(os.path.join(save_dir, 'test_results_compute_units.npy'), data)
    
    # 打印总结
    print("\n===== 总结 =====")
    print("\nRandom策略:")
    print(f"计算单元数: {compute_units_list}")
    print(f"平均奖励: {random_reward_list}")
    print(f"成功率: {random_success_rate_list}")
    print(f"PoT奖励: {random_pot_rewards_list}")
    print(f"感知功率: {random_p_sens_list}")
    print(f"LLM能量: {random_e_llm_list}")
    print(f"传输功率: {random_p_tx_list}")
    print(f"LLM准确率: {random_llm_accuracy_list}")
    print(f"CRB: {random_crb_list}")
    
    print("\nGreedy策略:")
    print(f"计算单元数: {compute_units_list}")
    print(f"平均奖励: {greedy_reward_list}")
    print(f"成功率: {greedy_success_rate_list}")
    print(f"PoT奖励: {greedy_pot_rewards_list}")
    print(f"感知功率: {greedy_p_sens_list}")
    print(f"LLM能量: {greedy_e_llm_list}")
    print(f"传输功率: {greedy_p_tx_list}")
    print(f"LLM准确率: {greedy_llm_accuracy_list}")
    print(f"CRB: {greedy_crb_list}")

def test_different_miner_rewards():
    """
    测试不同miner-reward值(0.3-0.7)下random和greedy策略的表现
    """
    # 确保保存目录存在
    save_dir = "/home/minrui/ISAC"
    os.makedirs(save_dir, exist_ok=True)
    
    miner_reward_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    # 创建指标列表
    miner_reward_list = []
    random_reward_list = []
    random_success_rate_list = []
    random_pot_rewards_list = []
    random_p_sens_list = []
    random_e_llm_list = []
    random_p_tx_list = []
    random_llm_accuracy_list = []
    random_crb_list = []
    
    greedy_reward_list = []
    greedy_success_rate_list = []
    greedy_pot_rewards_list = []
    greedy_p_sens_list = []
    greedy_e_llm_list = []
    greedy_p_tx_list = []
    greedy_llm_accuracy_list = []
    greedy_crb_list = []
    
    for miner_reward in miner_reward_values:
        print(f"\n===== 测试矿工奖励值 miner_reward = {miner_reward} =====")
        
        # 创建环境
        random_env = BlockchainISACEnvWithRequirements(N=3, U=5, max_episode_steps=100, p_max=1.0, miner_reward=miner_reward, seed=42)
        greedy_env = BlockchainISACEnvWithRequirements(N=3, U=5, max_episode_steps=100, p_max=1.0, miner_reward=miner_reward, seed=42)
        
        # 测试random策略
        print("\n测试random策略:")
        obs, info = random_env.reset()
        for i in range(100):
            action = random_env.action_space.sample()
            obs, reward, done, truncated, info = random_env.step(action)
            if done or truncated:
                obs, info = random_env.reset()
        
        # 测试greedy策略
        print("\n测试greedy策略:")
        obs, info = greedy_env.reset()
        for i in range(100):
            action = BlockchainISACEnvWithRequirements.greedy_algorithm(greedy_env, obs)
            obs, reward, done, truncated, info = greedy_env.step(action)
            if done or truncated:
                obs, info = greedy_env.reset()
        
        # 记录结果
        result = {
            'miner_reward': miner_reward,
            'random_reward': np.mean(random_env.all_reward_records),
            'random_success_rate': np.mean(random_env.all_success_records),
            'greedy_reward': np.mean(greedy_env.all_reward_records),
            'greedy_success_rate': np.mean(greedy_env.all_success_records),
            'random_pot_rewards': np.mean(random_env.all_pot_rewards_records),
            'random_p_sens': np.mean(random_env.all_p_sens_records),
            'random_e_llm': np.mean(random_env.all_e_llm_records),
            'random_p_tx': np.mean(random_env.all_p_tx_records),
            'random_llm_accuracy': np.mean(random_env.all_llm_accuracy_records),
            'random_crb': np.mean(random_env.all_new_crb_records),
            'greedy_pot_rewards': np.mean(greedy_env.all_pot_rewards_records),
            'greedy_p_sens': np.mean(greedy_env.all_p_sens_records),
            'greedy_e_llm': np.mean(greedy_env.all_e_llm_records),
            'greedy_p_tx': np.mean(greedy_env.all_p_tx_records),
            'greedy_llm_accuracy': np.mean(greedy_env.all_llm_accuracy_records),
            'greedy_crb': np.mean(greedy_env.all_new_crb_records)
        }
        results.append(result)
        
        # 添加到指标列表
        miner_reward_list.append(miner_reward)
        random_reward_list.append(result['random_reward'])
        random_success_rate_list.append(result['random_success_rate'])
        random_pot_rewards_list.append(result['random_pot_rewards'])
        random_p_sens_list.append(result['random_p_sens'])
        random_e_llm_list.append(result['random_e_llm'])
        random_p_tx_list.append(result['random_p_tx'])
        random_llm_accuracy_list.append(result['random_llm_accuracy'])
        random_crb_list.append(result['random_crb'])
        
        greedy_reward_list.append(result['greedy_reward'])
        greedy_success_rate_list.append(result['greedy_success_rate'])
        greedy_pot_rewards_list.append(result['greedy_pot_rewards'])
        greedy_p_sens_list.append(result['greedy_p_sens'])
        greedy_e_llm_list.append(result['greedy_e_llm'])
        greedy_p_tx_list.append(result['greedy_p_tx'])
        greedy_llm_accuracy_list.append(result['greedy_llm_accuracy'])
        greedy_crb_list.append(result['greedy_crb'])
        
        random_env.close()
        greedy_env.close()
    
    # 保存数据到文件
    data = {
        'miner_reward': miner_reward_list,
        'random_reward': random_reward_list,
        'random_success_rate': random_success_rate_list,
        'greedy_reward': greedy_reward_list,
        'greedy_success_rate': greedy_success_rate_list,
        'random_pot_rewards': random_pot_rewards_list,
        'random_p_sens': random_p_sens_list,
        'random_e_llm': random_e_llm_list,
        'random_p_tx': random_p_tx_list,
        'random_llm_accuracy': random_llm_accuracy_list,
        'random_crb': random_crb_list,
        'greedy_pot_rewards': greedy_pot_rewards_list,
        'greedy_p_sens': greedy_p_sens_list,
        'greedy_e_llm': greedy_e_llm_list,
        'greedy_p_tx': greedy_p_tx_list,
        'greedy_llm_accuracy': greedy_llm_accuracy_list,
        'greedy_crb': greedy_crb_list
    }
    np.save(os.path.join(save_dir, 'test_results_miner_rewards.npy'), data)
    
    # 打印总结
    print("\n===== 总结 =====")
    print("\nRandom策略:")
    print(f"矿工奖励值: {miner_reward_list}")
    print(f"平均奖励: {random_reward_list}")
    print(f"成功率: {random_success_rate_list}")
    print(f"PoT奖励: {random_pot_rewards_list}")
    print(f"感知功率: {random_p_sens_list}")
    print(f"LLM能量: {random_e_llm_list}")
    print(f"传输功率: {random_p_tx_list}")
    print(f"LLM准确率: {random_llm_accuracy_list}")
    print(f"CRB: {random_crb_list}")
    
    print("\nGreedy策略:")
    print(f"矿工奖励值: {miner_reward_list}")
    print(f"平均奖励: {greedy_reward_list}")
    print(f"成功率: {greedy_success_rate_list}")
    print(f"PoT奖励: {greedy_pot_rewards_list}")
    print(f"感知功率: {greedy_p_sens_list}")
    print(f"LLM能量: {greedy_e_llm_list}")
    print(f"传输功率: {greedy_p_tx_list}")
    print(f"LLM准确率: {greedy_llm_accuracy_list}")
    print(f"CRB: {greedy_crb_list}")

if __name__ == "__main__":
    print("===== 测试不同节点数量 N =====")
    test_different_N()
    # print("\n\n===== 测试不同目标数量 U =====")
    # test_different_U()
    # print("\n\n===== 测试不同最大计算单元数 =====")
    # test_different_compute_units()
    # print("\n\n===== 测试不同矿工奖励值 =====")
    # test_different_miner_rewards()
