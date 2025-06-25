import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

from utils.data_utils import create_unconditional_olympic_rings
from models.flow_matching import UnconditionalFlowMatching
from torch.distributions import MultivariateNormal

def plot_loss_curve(train_losses, save_path=None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_progression(model, save_dir, n_samples=1000):
    """Plot distribution progression at different time points."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample initial points
    base_dist = MultivariateNormal(
        loc=torch.zeros(2),
        covariance_matrix=torch.eye(2)
    )
    y_0 = base_dist.sample((n_samples,))
    
    # Plot at different time points
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # Compute vector field
        v = model.forward(y_0, t)
        
        # Update points
        y = y_0 + v * t
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y[:, 0].detach().cpu().numpy(), 
                   y[:, 1].detach().cpu().numpy(), 
                   s=1, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Distribution at t = {t:.1f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.savefig(f'{save_dir}/progression_t_{t:.1f}.png')
        plt.close()

def plot_trajectories(model, n_points=10, save_path=None):
    """Plot trajectories of points through the flow."""
    # Sample initial points
    base_dist = MultivariateNormal(
        loc=torch.zeros(2),
        covariance_matrix=torch.eye(2)
    )
    y_0 = base_dist.sample((n_points,))
    
    # Track points through time
    points = [y_0.detach().cpu().numpy()]
    y = y_0
    dt = 1/1000
    
    t = 0.0
    while t < 1.0:
        v = model.forward(y, t)
        y = y + v * dt
        t += dt
        points.append(y.detach().cpu().numpy())
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    points = np.array(points)  # Shape: (n_steps, n_points, 2)
    
    # Create colormap for time
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    for i in range(n_points):
        plt.plot(points[:, i, 0], points[:, i, 1], 'o-', alpha=0.5, label=f'Point {i+1}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=plt.gca(), label='Time t')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Sampling Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_sample_distributions(model, save_dir, n_samples=1000):
    """Plot sample distributions for different time steps."""
    os.makedirs(save_dir, exist_ok=True)
    
    for dt in [0.002, 0.02, 0.05, 0.1, 0.2]:
        samples = model.sample(n_samples, dt=dt)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(samples[:, 0].detach().cpu().numpy(),
                   samples[:, 1].detach().cpu().numpy(),
                   s=1, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Sample Distribution (Δt = {dt})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.savefig(f'{save_dir}/samples_dt_{dt}.png')
        plt.close()

def plot_reverse_trajectories(model, points, save_path=None):
    """Plot reverse flow trajectories for specific points."""
    # Track points through reverse flow
    points_tracked = [points.detach().cpu().numpy()]
    y = points
    dt = 1/1000
    
    t = 1.0
    while t > 0.0:
        v = model.forward(y, t)
        y = y - v * dt
        t -= dt
        points_tracked.append(y.detach().cpu().numpy())
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    points_tracked = np.array(points_tracked)  # Shape: (n_steps, n_points, 2)
    
    # Create colormap for time
    colors = plt.cm.viridis(np.linspace(0, 1, len(points_tracked)))
    
    for i in range(len(points)):
        plt.plot(points_tracked[:, i, 0], points_tracked[:, i, 1], 'o-', alpha=0.5, label=f'Point {i+1}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=plt.gca(), label='Time t')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Reverse Flow Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_flow_matching(
    n_points=250000,
    batch_size=128,
    n_epochs=20,
    learning_rate=1e-3,
    device='cpu',
    hidden_dim=64,
    num_layers=4
):
    """
    Train the unconditional flow matching model.
    
    Args:
        n_points (int): Number of data points
        batch_size (int): Batch size
        n_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to train on
        hidden_dim (int): Width of hidden layers
        num_layers (int): Number of hidden layers
    """
    # Create directories for plots
    plots_dir = "flow_matching_plots"
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize model
    model = UnconditionalFlowMatching(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    
    # Get data
    data = create_unconditional_olympic_rings(n_points)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    train_losses = []
    
    print("\nStarting training...")
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            # Sample initial points from base distribution
            y_0 = model.base_distribution.sample((batch.shape[0],)).to(device)
            
            # Ensure batch is float32
            batch = batch.float()
            
            # Compute loss
            loss = model.loss((y_0, batch))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        # Save average loss
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f'\nEpoch {epoch+1}/{n_epochs} Summary:')
        print(f'  Training Loss: {avg_loss:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print("\nTraining completed!")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses
    }, 'flow_matching_model.pt')
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Q1: Plot loss curve
    plot_loss_curve(train_losses, save_path=f'{plots_dir}/loss_curve.png')
    
    # Q2: Plot progression
    plot_progression(model, save_dir=f'{plots_dir}/progression')
    
    # Q3: Plot trajectories
    plot_trajectories(model, save_path=f'{plots_dir}/trajectories.png')
    
    # Q4: Plot sample distributions
    plot_sample_distributions(model, save_dir=f'{plots_dir}/samples')
    
    # Q5: Plot reverse trajectories
    points = torch.tensor([
        [0.0, 0.0],    # Center
        [0.5, 0.5],    # Inside ring
        [-0.5, 0.5],   # Inside ring
        [2.0, 2.0],    # Outside
        [-2.0, -2.0]   # Outside
    ], device=device)
    plot_reverse_trajectories(model, points, save_path=f'{plots_dir}/reverse_trajectories.png')
    
    return model, train_losses

if __name__ == '__main__':
    # Train the model
    model, train_losses = train_flow_matching() 