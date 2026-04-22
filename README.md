# ISAC-PPO Project

This repository contains the simulation code for the paper:  
**"A DePIN Architecture for Large Language Model-Driven Integrated Sensing and Communications"**

## Overview

A decentralized Integrated Sensing and Communication (ISAC) framework that combines:
- Permissioned blockchain with oracle contracts for semantic validation of LLM-generated outputs
- Proof-of-Code (PoC) consensus mechanism with stake-weighted validator selection
- PPO-based deep reinforcement learning for adaptive incentive and energy allocation

## File Structure

- `ISACEnv.py` — ISAC environment definition (state, action, reward, CRB, SINR)
- `ppo_trainer.py` — PPO training loop based on the Tianshou framework
- `data.py` — Data processing module
- `extract_data.py` — Extracts logs from training runs
- `plot_rewards.py` — Reward curve visualization
- `plot_convergence.py` — Convergence analysis plots
- `plot_comparison.py` — PPO vs. Greedy vs. Random comparison
- `visualize_results.py` — Final result visualization

## Requirements

All dependencies are listed in `requirements.txt`. Tested on Python 3.9.

Install via:

```bash
pip install -r requirements.txt
```

## Dataset

Multimodal sensory inputs (GPS coordinates and RGB images) are drawn from the publicly available **DeepSense 6G** dataset, Scenarios 32–34:  
https://www.deepsense6g.net/

## Usage

```bash
# Train the PPO agent
python ppo_trainer.py

# Plot reward curves
python plot_rewards.py

# Plot convergence analysis
python plot_convergence.py

# Compare PPO against Greedy and Random baselines
python plot_comparison.py
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite our paper (citation details will be updated upon publication).
