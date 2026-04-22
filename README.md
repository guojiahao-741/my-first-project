# ISAC-PPO Project

Integrated Sensing and Communication (ISAC) system with PPO-based reinforcement learning.

## File Structure

- `ISACEnv.py` — ISAC environment definition
- `data.py` — Data processing module
- `extract_data.py` — Data extraction utility
- `ppo_trainer.py` — PPO training implementation
- `plot_comparison.py` — Comparison plots
- `plot_convergence.py` — Convergence plots
- `plot_rewards.py` — Reward plots
- `visualize_results.py` — Results visualization

## Requirements

- Python 3.9+
- numpy
- matplotlib
- torch (for PPO)

## Usage

Train the PPO agent:

```bash
python ppo_trainer.py
```

Visualize results:

```bash
python visualize_results.py
```

## Notes

Log files (`log*.txt`) and test result files (`test_results_*.npy`) are generated during training and are not included in this repository.

## License

MIT