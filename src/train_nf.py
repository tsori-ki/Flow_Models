import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import shutil
from sklearn.model_selection import train_test_split

from utils.data_utils import get_unconditional_data
from models.normalizing_flow import NormalizingFlowModel
from utils.plotting_utils import (
    plot_sampling_results,
    plot_sampling_over_layers,
    plot_sampling_trajectories,
    plot_inverse_trajectories,
    compute_point_probabilities,
    plot_loss_components
)

def train_normalizing_flow(
    n_points=250000,
    batch_size=128,
    n_epochs=20,
    learning_rate=1e-3,
    device='cpu',
    n_layers=15,
    D=2,
    save_dir='plots/training'
):
    """
    Train the Normalizing Flow model.
    
    Args:
        n_points (int): Number of data points to generate
        batch_size (int): Batch size for training
        n_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on
        n_layers (int): Number of flow layers
        D (int): Input dimension
        save_dir (str): Directory to save training plots
        
    Returns:
        tuple: (model, training_losses, validation_losses, val_log_det_components, val_log_pz_components)
    """
    # Create all necessary directories
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'epoch_samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'layers'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'trajectories'), exist_ok=True)
    
    print(f"\nInitializing training with:")
    print(f"- Number of points: {n_points}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {n_epochs}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Device: {device}")
    print(f"- Number of layers: {n_layers}")
    print(f"- Input dimension: {D}")
    print(f"- Training plots will be saved to: {save_dir}\n")
    
    # Initialize model and move to device
    print("Initializing model...")
    model = NormalizingFlowModel(n_layers=n_layers, D=D).to(device)
    
    # Initialize base distribution (standard normal)
    print("Initializing base distribution...")
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    
    # Get data
    print("Loading data...")
    data = get_unconditional_data(n_points, device=device)
    
    # Split into train and validation sets (90/10 split)
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer and scheduler
    print("Initializing optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_log_det_components = []
    val_log_pz_components = []
    
    print("\nStarting training...")
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        # Training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            z, log_det = model.inverse(batch)
            log_pz = base_distribution.log_prob(z)
            
            # Compute loss
            loss = -(log_pz + log_det).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        # Compute average training loss
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_epoch_losses = []
        val_epoch_logdet = []
        val_epoch_logpz = []
        
        print(f"\nRunning validation for epoch {epoch+1}...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = batch.to(device)
                
                # Forward pass
                z, log_det = model.inverse(batch)
                log_pz = base_distribution.log_prob(z)
                
                # Compute loss
                loss = -(log_pz + log_det).mean()
                val_epoch_losses.append(loss.item())
                val_epoch_logdet.append(log_det.mean().item())
                val_epoch_logpz.append(log_pz.mean().item())
        
        # Compute average validation loss
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        val_log_det_components.append(np.mean(val_epoch_logdet))
        val_log_pz_components.append(np.mean(val_epoch_logpz))
        
        # Save model after each epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_log_det': val_log_det_components,
            'val_log_pz': val_log_pz_components,
            'epoch': epoch
        }, os.path.join(save_dir, 'best_model.pt'))
        
        # Save sampled point distribution after each epoch
        print(f"\nSaving sampled point distribution for epoch {epoch + 1}...")
        plot_sampling_results(
            model, 
            n_samples=1000, 
            n_seeds=1,  # Single seed for epoch-wise sampling
            save_dir=os.path.join(save_dir, 'epoch_samples'),
            filename=f"epoch_{epoch + 1}.png"
        )
        
        print(f'\nEpoch {epoch+1}/{n_epochs} Summary:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print("\nTraining completed!")
    
    # Generate final plots
    print("\nGenerating final plots...")
    generate_plots(model, train_losses, val_losses, val_log_det_components, val_log_pz_components, device, save_dir)
    
    return model, train_losses, val_losses, val_log_det_components, val_log_pz_components

def generate_plots(model, train_losses, val_losses, val_log_det, val_log_pz, device, save_dir='plots'):
    """Generate all required plots using the trained model."""
    # Create output directory for plots
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss components
    print('Generating loss plots...')
    plot_loss_components(
        train_losses,
        val_losses,
        val_log_det,
        val_log_pz,
        save_path=os.path.join(save_dir, 'loss_components.png')
    )
    
    # Generate samples with different seeds
    print('Generating samples with different seeds...')
    plot_sampling_results(
        model,
        n_samples=1000,
        n_seeds=3,
        save_dir=os.path.join(save_dir, 'samples')
    )
    
    # Plot sampling over layers
    print('Generating sampling over layers plots...')
    plot_sampling_over_layers(
        model,
        n_samples=1000,
        save_dir=os.path.join(save_dir, 'layers')
    )
    
    # Plot sampling trajectories
    print('Generating sampling trajectories...')
    plot_sampling_trajectories(
        model,
        n_points=10,
        save_path=os.path.join(save_dir, 'trajectories', 'sampling_trajectories.png')
    )
    
    # Plot inverse trajectories and compute probabilities
    print('Generating inverse trajectories and computing probabilities...')
    
    # Get training data for normalization
    data = get_unconditional_data(250000, device='cpu').cpu().numpy()
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    print(f"Data mean: {mean}, std: {std}")
    # Olympic ring parameters
    centers = np.array([
        [0, 0],    # blue
        [2, 0],    # black
        [4, 0],    # red
        [1, -1],   # yellow
        [3, -1]    # green
    ])
    radius = 1
    # Pick 3 points on the edge of 3 different rings at different angles
    thetas = [0, np.pi/2, np.pi]  # angles for variety
    edge_points = np.array([
        centers[0] + radius * np.array([np.cos(thetas[0]), np.sin(thetas[0])]),
        centers[1] + radius * np.array([np.cos(thetas[1]), np.sin(thetas[1])]),
        centers[2] + radius * np.array([np.cos(thetas[2]), np.sin(thetas[2])]),
    ])
    # 2 points far outside
    outside_points = np.array([[5, 5], [-5, -5]])
    # Normalize all points
    all_points = np.vstack([edge_points, outside_points])
    all_points_norm = (all_points - mean) / std
    print(f"Edge points (unnormalized):\n{edge_points}")
    print(f"Normalized points for inverse plot:\n{all_points_norm}")
    points = torch.tensor(all_points_norm, dtype=torch.float32, device=device)
    
    # Plot inverse trajectories
    plot_inverse_trajectories(
        model,
        points,
        save_path=os.path.join(save_dir, 'trajectories', 'inverse_trajectories.png')
    )
    
    # Plot inverse trajectories for points sampled from the model
    print('\nGenerating inverse trajectories for points sampled from the model...')
    # Initialize base distribution
    D = model.D # Get dimension from model
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    # Sample points from the base distribution
    z_samples = base_distribution.sample((5,))
    # Transform to data space using model.forward
    x_samples, _ = model.forward(z_samples) # These are in normalized data space
    print(f"Points sampled from model (normalized) for inverse plot:\n{x_samples.detach().cpu().numpy()}")
    plot_inverse_trajectories(
        model,
        x_samples, # Pass these directly as they are already normalized
        save_path=os.path.join(save_dir, 'trajectories', 'inverse_trajectories_from_samples.png')
    )

    # Compute probabilities (for the original hand-picked points, if desired, or remove/comment out)
    # print('\nProbability components for selected points:')
    # ... (existing probability computation for hand-picked points) ...

def main():
    parser = argparse.ArgumentParser(description='Train and visualize Normalizing Flow model')
    parser.add_argument('--mode', type=str, choices=['train', 'load'], default='train',
                      help='Mode: train a new model or load an existing one')
    parser.add_argument('--load_path', type=str, default='checkpoints/best_model.pt',
                      help='Path to load pre-trained model weights')
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.pt',
                      help='Path to save the trained model weights')
    parser.add_argument('--n_points', type=int, default=250000,
                      help='Number of data points to generate')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate for optimizer')
    parser.add_argument('--n_layers', type=int, default=15,
                      help='Number of flow layers')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    if args.mode == 'load':
        # Load pre-trained model
        print(f'Loading model from {args.load_path}...')
        checkpoint = torch.load(args.load_path, map_location=device)
        model = NormalizingFlowModel(n_layers=args.n_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_log_det = checkpoint.get('val_log_det', None)
        val_log_pz = checkpoint.get('val_log_pz', None)
        
        # Generate plots
        generate_plots(model, train_losses, val_losses, val_log_det, val_log_pz, device)
    else:
        # Train the model
        print('Training Normalizing Flow model...')
        model, train_losses, val_losses, val_log_det, val_log_pz = train_normalizing_flow(
            n_points=args.n_points,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            device=device,
            n_layers=args.n_layers
        )
        
        # Save the model
        print(f'Saving model to {args.save_path}...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_log_det': val_log_det,
            'val_log_pz': val_log_pz
        }, args.save_path)

if __name__ == '__main__':
    main() 