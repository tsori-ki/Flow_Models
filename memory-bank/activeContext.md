# Active Context

## Current Focus
1. Implementing and verifying all required visualizations for both NF and FM models
2. Ensuring proper model architectures match requirements
3. Setting up proper training and evaluation pipelines

## Recent Changes
1. Organized project structure with src/, models/, utils/ directories
2. Implemented basic NF and FM model architectures
3. Set up data utilities and plotting functions
4. Created training scripts for both models

## Next Steps
1. **Model Verification**
   - [ ] Verify NF model has exactly 15 layers with correct architecture
   - [ ] Verify FM model has correct number of FC layers and widths
   - [ ] Test both models' forward and inverse operations

2. **Visualization Implementation**
   - [ ] Implement NF Q1-Q5 plots:
     - [ ] Loss components over epochs
     - [ ] Sampling results with different seeds
     - [ ] Layer-wise distributions
     - [ ] Forward trajectories
     - [ ] Inverse trajectories with probability computation
   - [ ] Implement FM Q1-Q5 plots:
     - [ ] MSE loss over training
     - [ ] Flow progression snapshots
     - [ ] Sampling trajectories
     - [ ] Delta-t comparison
     - [ ] Reverse trajectories

3. **Training Pipeline**
   - [ ] Verify hyperparameters match requirements
   - [ ] Implement proper checkpointing
   - [ ] Add evaluation metrics
   - [ ] Test both training and inference modes

4. **Documentation**
   - [ ] Update memory bank files
   - [ ] Add docstrings to all functions
   - [ ] Create usage examples
   - [ ] Document visualization requirements

## Active Decisions
1. Using LeakyReLU for all activations
2. Implementing both unconditional and conditional FM variants
3. Saving all plots to organized subdirectories
4. Using cosine annealing for learning rate scheduling

## Current Considerations
1. Need to ensure efficient visualization code
2. Need to verify model architectures match requirements exactly
3. Need to implement proper evaluation metrics
4. Need to ensure reproducible training process 