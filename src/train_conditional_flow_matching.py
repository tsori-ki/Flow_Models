import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

from utils.data_utils import create_olympic_rings
from flow_matching import ConditionalFlowMatching
from torch.distributions import MultivariateNormal

def plot_input_data(data, labels, save_path=None):
    """Plot input data colored by class."""
    plt.figure(figsize=(10, 10))
    for i in range(5):  # 5 classes
        mask = labels == i
        plt.scatter(data[mask, 0], data[mask, 1], s=1, alpha=0.5, label=f'Class {i+1}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Input Data (Colored by Class)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_class_trajectories(model, save_path=None):
    """Plot one trajectory per class."""
    # Sample initial points
    base_dist = MultivariateNormal(
        loc=torch.zeros(2),
        covariance_matrix=torch.eye(2)
    )
    y_0 = base_dist.sample((5,))  # One point per class
    
    # Track points through time
    points = [y_0.detach().cpu().numpy()]
    y = y_0
    dt = 1/1000
    
    t = 0.0
    while t < 1.0:
        # Create class indices
        class_ids = torch.arange(5, device=y.device)
        
        # Compute vector field for each class
        v = model.forward(y, t, class_ids)
        
        # Update points
        y = y + v * dt
        t += dt
        points.append(y.detach().cpu().numpy())
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    points = np.array(points)  # Shape: (n_steps, n_classes, 2)
    
    # Create colormap for time
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    for i in range(5):  # 5 classes
        plt.plot(points[:, i, 0], points[:, i, 1], 'o-', alpha=0.5, label=f'Class {i+1}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=plt.gca(), label='Time t')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Class Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_generated_samples(model, n_samples=3000, save_path=None):
    """Plot generated samples colored by class."""
    # Generate samples for each class
    all_samples = []
    all_labels = []
    
    for class_id in range(5):
        samples = model.sample(n_samples // 5, class_id)
        all_samples.append(samples)
        all_labels.extend([class_id] * (n_samples // 5))
    
    # Concatenate samples
    samples = torch.cat(all_samples, dim=0)
    labels = torch.tensor(all_labels)
    
    # Plot
    plt.figure(figsize=(10, 10))
    for i in range(5):
        mask = labels == i
        plt.scatter(samples[mask, 0].detach().cpu().numpy(),
                   samples[mask, 1].detach().cpu().numpy(),
                   s=1, alpha=0.5, label=f'Class {i+1}')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Generated Samples (Colored by Class)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_conditional_flow_matching(
    n_points=250000,
    batch_size=128,
    n_epochs=20,
    learning_rate=1e-3,
    device='cpu',
    hidden_dim=64,
    num_layers=4,
    num_classes=5,
    emb_dim=32
):
    """
    Train the conditional flow matching model.
    
    Args:
        n_points (int): Number of data points
        batch_size (int): Batch size
        n_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to train on
        hidden_dim (int): Width of hidden layers
        num_layers (int): Number of hidden layers
        num_classes (int): Number of classes
        emb_dim (int): Dimension of class embeddings
    """
    # Create directories for plots
    plots_dir = "conditional_flow_matching_plots"
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize model
    model = ConditionalFlowMatching(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        emb_dim=emb_dim
    ).to(device)
    
    # Get data
    data, labels = create_olympic_rings(n_points, return_class=True)
    dataset = torch.utils.data.TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Plot input data
    plot_input_data(data.cpu().numpy(), labels.cpu().numpy(),
                   save_path=f'{plots_dir}/input_data.png')
    
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
        for batch, class_ids in pbar:
            # Sample initial points from base distribution
            y_0 = model.base_distribution.sample((batch.shape[0],)).to(device)
            
            # Compute loss
            loss = model.loss((y_0, batch, class_ids))
            
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
    }, 'conditional_flow_matching_model.pt')
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Q2: Plot class trajectories
    plot_class_trajectories(model, save_path=f'{plots_dir}/class_trajectories.png')
    
    # Q3: Plot generated samples
    plot_generated_samples(model, save_path=f'{plots_dir}/generated_samples.png')
    
    return model, train_losses

if __name__ == '__main__':
    # Train the model
    model, train_losses = train_conditional_flow_matching() 