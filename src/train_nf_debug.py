import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import shutil
import matplotlib.pyplot as plt
from pathlib import Path

from utils.data_utils import get_unconditional_data, create_data_loader
from models.normalizing_flow import NormalizingFlowModel
from utils.plotting_utils import (
    plot_loss_components,
    plot_sampling_results,
    plot_sampling_over_layers,
    plot_sampling_trajectories
)

def plot_samples(samples, title, save_path):
    """Plot generated samples."""
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    plt.title(title)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_trajectories(model, z, save_path):
    """Plot trajectories of points through the flow."""
    plt.figure(figsize=(8, 8))
    
    # Get intermediate outputs
    outputs = model.get_layer_outputs(z)
    
    # Plot trajectory for each point
    for i in range(z.shape[0]):
        points = torch.stack([out[i] for out in outputs])
        plt.plot(points[:, 0].cpu(), points[:, 1].cpu(), 'b-', alpha=0.3)
        plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), c=range(len(points)), 
                   cmap='viridis', s=10, alpha=0.5)
    
    plt.title('Point Trajectories Through Flow')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.colorbar(label='Layer Index')
    plt.savefig(save_path)
    plt.close()

def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_normalizing_flow(n_points=250000, batch_size=128, n_epochs=7, lr=0.001, 
                         n_layers=15, D=2, debug=True):
    """
    Train a normalizing flow model on the Olympic rings data.
    
    Args:
        n_points (int): Number of data points to generate
        batch_size (int): Batch size for training
        n_epochs (int): Number of training epochs
        lr (float): Learning rate
        n_layers (int): Number of flow layers
        D (int): Input dimension
        debug (bool): Whether to run in debug mode (fewer epochs, more visualizations)
    """
    # Set up device
    device = torch.device('cpu')
    print(f"\nInitializing training with:")
    print(f"- Number of points: {n_points}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {n_epochs}")
    print(f"- Learning rate: {lr}")
    print(f"- Device: {device}")
    print(f"- Number of layers: {n_layers}")
    print(f"- Input dimension: {D}")
    
    # Create debug directory if needed
    if debug:
        debug_dir = Path("debug_plots")
        debug_dir.mkdir(exist_ok=True)
        print(f"- Debug plots will be saved to: {debug_dir}")
    
    # Initialize model
    print("Initializing base distribution...")
    model = NormalizingFlowModel(n_layers=n_layers, D=D).to(device)
    
    # Load data
    print("Loading data...")
    data = get_unconditional_data(n_points)
    
    # Split into train and validation
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer
    print("Initializing optimizer (scheduler disabled)...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            
            # Compute loss: -log p(x) = -log p_z(z) - log|det J|
            z, log_det_J = model.inverse(x)
            log_prob_z = model.base_dist.log_prob(z)
            loss = -(log_prob_z + log_det_J).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                z, log_det_J = model.inverse(x)
                log_prob_z = model.base_dist.log_prob(z)
                loss = -(log_prob_z + log_det_J).mean()
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}")
        
        # Generate and plot samples
        if debug:
            # Sample from model with different seeds
            for seed in [42, 123, 456]:
                samples = model.sample(1000, seed=seed)
                plot_samples(samples.cpu().numpy(),
                           f'Samples (Epoch {epoch+1}, Seed {seed})',
                           debug_dir / f'samples_epoch{epoch+1}_seed{seed}.png')
    
    # Plot losses
    if debug:
        plot_losses(train_losses, val_losses,
                   debug_dir / 'losses.png')
        
        # Plot trajectories for 10 random points
        z = model.base_dist.sample((10,))
        plot_trajectories(model, z,
                         debug_dir / 'trajectories.png')
        
        # Plot layer-wise transformations for 1000 points
        z = model.base_dist.sample((1000,))
        outputs = model.get_layer_outputs(z)
        for i, out in enumerate(outputs[::3]):  # Plot every 3rd layer
            plot_samples(out.cpu().numpy(),
                        f'Distribution after Layer {i*3}',
                        debug_dir / f'layer_{i*3}.png')
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    train_normalizing_flow()