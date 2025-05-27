# Normalizing Flow Models for Olympic Rings Data

This project implements Normalizing Flow models for generating and analyzing 2D Olympic rings data.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

This will:
- Train the Normalizing Flow model
- Generate all required plots in the `plots` directory
- Print probability components for selected points

## Project Structure

- `data_utils.py`: Data loading and preparation utilities
- `normalizing_flow.py`: Core model components (AffineCouplingLayer, PermutationLayer, NormalizingFlowModel)
- `train_nf.py`: Training loop implementation
- `plotting_utils.py`: Visualization functions
- `main.py`: Main script to run experiments
- `create_data.py`: Data generation utilities (provided)

## Output

The script will generate the following plots in the `plots` directory:
1. `loss_components.png`: Training and validation losses
2. `samples_seed_*.png`: Generated samples with different random seeds
3. `layer_*.png`: Distribution of samples after each N layers
4. `sampling_trajectories.png`: 2D trajectories of points through layers
5. `inverse_trajectories.png`: Inverse trajectories of selected points

The trained model will be saved as `normalizing_flow_model.pt`. 