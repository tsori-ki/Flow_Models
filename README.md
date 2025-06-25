# Flow-Based Generative Models – Normalizing Flows & Flow Matching

This project implements two flow-based generative modeling approaches — **Normalizing Flows** and **Flow Matching** — to learn complex 2D data distributions. The models are trained to reproduce and analyze data shaped like the Olympic rings.

---

## Normalizing Flows

Normalizing Flows learn invertible transformations that map a simple base distribution into a complex target distribution.

Key details:
- Architecture: 15 affine coupling layers with interleaved permutations
- Optimization: Maximum Likelihood using change-of-variable formula
- Outputs: Loss plots, per-layer sample distributions, trajectory visualizations
- Sampling and inversion visualizations demonstrate interpretability
- Based on the Real NVP architecture by Dinh et al. (2017). [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

---

## Flow Matching

Flow Matching learns continuous-time transformation fields between a prior and target distribution.

Key details:
- Implements both unconditional and class-conditional flows
- Models a time-dependent vector field using MLPs
- Sampling via Euler integration, forward and reverse
- Evaluates flow consistency and trajectory smoothness
- Implements the Flow Matching approach introduced by Lipman et al. (2022). [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)


---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main.py
```

This will:
- Train the Normalizing Flow model
- Generate plots in the `plots/` directory
- Print log-probability components for selected points

---

## Key Files

- `data_utils.py`: Data generation and loading (Olympic rings)
- `normalizing_flow.py`: Core flow model components
- `train_nf.py`: Training loop for Normalizing Flows
- `flow_matching.py`: Flow Matching architecture and training
- `plotting_utils.py`: Visualization utilities
- `main.py`: Controller script to run experiments

---

## Output

All result plots are saved to the `plots/` folder:

- `loss_components.png`: Loss curves and components
- `samples_seed_*.png`: Sampled outputs with different random seeds
- `layer_*.png`: Visualization of sample transformation over layers
- `sampling_trajectories.png`: Forward sampling as 2D trajectories
- `inverse_trajectories.png`: Inverse flow over selected points
- `conditional_sampling.png`: Class-aware sampling from flow matching

Trained models are saved as `.pt` files (e.g., `normalizing_flow_model.pt`).

---

## Technologies

- Python 3.10 / PyTorch
- Matplotlib, Numpy
- Cosine Learning Rate Scheduler

---

## Highlights

- All implementations built fully from scratch using PyTorch
- 2D problem setup enables interpretable training and visualization
- Separate analysis of sampling consistency, likelihood estimation, and flow-based transformation quality
```