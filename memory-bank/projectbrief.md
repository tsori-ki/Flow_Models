# Project Brief: Flow Models Implementation

## Core Requirements
1. Implement two types of flow models:
   - Normalizing Flow (NF)
   - Flow Matching (FM) - both unconditional and conditional variants

2. Model Architectures:
   - NF: 15-layer architecture with AffineCoupling and Permutation layers
   - FM: Vector Field Network with 4-5 FC layers and conditional support

3. Data:
   - 2D Olympic rings dataset
   - 250,000 data points
   - Both unconditional and conditional variants

4. Training Requirements:
   - Batch size: 128
   - Epochs: 20 (7 for debugging)
   - Optimizer: Adam with lr=1e-3
   - Scheduler: CosineAnnealingLR

5. Visualization Requirements:
   - NF: 5 types of plots (Q1-Q5)
   - FM: 5 types of plots (Q1-Q5)
   - Trajectory visualizations
   - Sampling results
   - Loss components

## Project Goals
1. Create clean, modular implementations of both flow models
2. Achieve good sampling quality for the Olympic rings dataset
3. Provide comprehensive visualization tools
4. Support both training and inference modes
5. Maintain clear documentation and code organization

## Success Criteria
1. Models successfully learn the Olympic rings distribution
2. All required visualizations are implemented and working
3. Code is well-organized and documented
4. Training process is reproducible
5. Models can be saved and loaded for inference 