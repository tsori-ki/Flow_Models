# Technical Context

## Technologies Used
- Python 3.x
- PyTorch
- NumPy
- Matplotlib for visualization

## Project Structure
```
.
├── src/
│   ├── models/
│   │   ├── normalizing_flow.py    # NF model implementation
│   │   └── flow_matching.py       # FM model implementation
│   ├── utils/
│   │   ├── data_utils.py         # Data generation and handling
│   │   ├── plotting_utils.py     # Visualization functions
│   │   └── create_data.py        # Olympic rings data generation
│   ├── train_nf.py               # NF training script
│   ├── train_flow_matching.py    # Unconditional FM training
│   └── train_conditional_flow_matching.py  # Conditional FM training
├── plots/                        # Generated plots
├── checkpoints/                  # Saved model checkpoints
└── memory-bank/                  # Project documentation
```

## Model Architectures

### Normalizing Flow
1. AffineCouplingLayer:
   - Input: 2D tensor [batch_size, 2]
   - Neural network f: 5 linear layers (hidden size=8)
   - LeakyReLU activation
   - Output: log_s and b for affine transformation

2. PermutationLayer:
   - Simple dimension swapping
   - Determinant = 1

3. NormalizingFlowModel:
   - 15 AffineCoupling layers
   - Interleaved Permutation layers
   - Base distribution: MultivariateNormal

### Flow Matching
1. VectorFieldNet:
   - 4-5 FC layers (width ≥ 64)
   - LeakyReLU activation
   - Conditional support via embedding
   - Input: (y, t, [class_label])
   - Output: 2D vector field

## Training Process
1. Data Generation:
   - 250,000 points
   - Olympic rings distribution
   - Train/val split

2. Optimization:
   - Adam optimizer (lr=1e-3)
   - Cosine annealing scheduler
   - Batch size: 128
   - 20 epochs (7 for debugging)

3. Checkpointing:
   - Save best model
   - Save training plots
   - Save sampling results

## Visualization Requirements
1. NF Plots (Q1-Q5):
   - Loss components
   - Sampling results
   - Layer-wise distributions
   - Forward trajectories
   - Inverse trajectories

2. FM Plots (Q1-Q5):
   - MSE loss
   - Flow progression
   - Sampling trajectories
   - Delta-t comparison
   - Reverse trajectories

## Data Generation
- The module `src/utils/create_data.py` is a core data generation module and should not be modified. It provides functions for generating Olympic rings data for both conditional and unconditional tasks. 