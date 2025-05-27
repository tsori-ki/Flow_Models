import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from models.normalizing_flow import AffineCouplingLayer

def plot_loss_components(train_losses, val_losses, log_det_components=None, log_pz_components=None, save_path=None):
    """
    Plot loss curves and (optionally) their two components.
    If log_det_components / log_pz_components are provided they are drawn in
    the same figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, label='Validation Loss', color='tab:red')
    plt.plot(train_losses, label='Training Loss', color='tab:orange', alpha=0.6)
    if log_det_components is not None and log_pz_components is not None:
        plt.plot(log_det_components, label='Mean log |det J|', color='tab:blue')
        plt.plot(log_pz_components, label='Mean log p(z)', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Components over Epochs')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_sampling_results(model, n_samples=1000, n_seeds=3, save_dir=None, filename=None):
    """
    Generate and plot samples from the model with different random seeds.
    
    Args:
        model (NormalizingFlowModel): Trained model
        n_samples (int): Number of samples to generate
        n_seeds (int): Number of different random seeds to use
        save_dir (str, optional): Directory to save the plots
        filename (str, optional): Custom filename for the plot
    """
    device = next(model.parameters()).device
    D = model.D
    
    # Initialize base distribution
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        
        # Generate samples from base distribution
        z = base_distribution.sample((n_samples,))
        # Transform to data space using forward
        x, _ = model.forward(z)
        
        # Convert to numpy for plotting (with detach)
        x_np = x.detach().cpu().numpy()
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(x_np[:, 0], x_np[:, 1], s=1, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Samples from Normalizing Flow (Seed {seed})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        if save_dir:
            if filename:
                plt.savefig(f'{save_dir}/{filename}')
            else:
                plt.savefig(f'{save_dir}/samples_seed_{seed}.png')
        plt.close()

def plot_sampling_over_layers(model, n_samples=1000, save_dir=None):
    """
    Plot the distribution of samples after each affine coupling layer, showing exactly 5 plots.
    
    Args:
        model (NormalizingFlowModel): Trained model
        n_samples (int): Number of samples to generate
        save_dir (str, optional): Directory to save the plots
    """
    device = next(model.parameters()).device
    D = model.D
    
    # Initialize base distribution
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    
    # Generate initial samples
    z = base_distribution.sample((n_samples,))
    
    # Plot initial distribution
    plt.figure(figsize=(8, 8))
    plt.scatter(z.detach().cpu().numpy()[:, 0], z.detach().cpu().numpy()[:, 1], s=1, alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Initial Distribution (Layer 0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/layer_0.png')
    plt.close()
    
    # Count affine coupling layers
    n_affine_layers = sum(1 for layer in model.layers if isinstance(layer, AffineCouplingLayer))
    plot_every = max(1, n_affine_layers // 4)  # This will give us 5 plots total (including initial)
    
    # Plot distribution after each affine coupling layer
    x = z
    affine_layer_count = 0
    
    for i, layer in enumerate(model.layers):
        x, _ = layer(x)  # Unpack the tuple, ignore log_det
        if isinstance(layer, AffineCouplingLayer):
            affine_layer_count += 1
            if affine_layer_count % plot_every == 0 or affine_layer_count == n_affine_layers:
                plt.figure(figsize=(8, 8))
                plt.scatter(x.detach().cpu().numpy()[:, 0], x.detach().cpu().numpy()[:, 1], s=1, alpha=0.5)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title(f'Distribution after Affine Layer {affine_layer_count}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid(True)
                if save_dir:
                    plt.savefig(f'{save_dir}/affine_layer_{affine_layer_count}.png')
                plt.close()

def plot_sampling_trajectories(model, n_points=10, save_path=None):
    """
    Plot the 2D trajectories of points as they pass through the model layers.
    
    Args:
        model (NormalizingFlowModel): Trained model
        n_points (int): Number of points to track
        save_path (str, optional): Path to save the plot
    """
    device = next(model.parameters()).device
    D = model.D
    
    # Initialize base distribution
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    
    # Generate initial points
    z = base_distribution.sample((n_points,))
    
    # Track points through layers
    points = [z.detach().cpu().numpy()]
    x = z
    for layer in model.layers:
        x, _ = layer(x)  # Unpack the tuple, ignore log_det
        points.append(x.detach().cpu().numpy())
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    points = np.array(points)  # Shape: (n_layers + 1, n_points, 2)
    
    # Create a colormap for the layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    for i in range(n_points):
        plt.plot(points[:, i, 0], points[:, i, 1], 'o-', alpha=0.5, label=f'Point {i+1}')
    
    # Add colorbar to show layer progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, len(points)-1))
    plt.colorbar(sm, ax=plt.gca(), label='Layer Index')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Sampling Trajectories through Layers')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_inverse_trajectories(model, points, save_path=None):
    """
    Plot the inverse trajectories of specific points through the model layers.
    
    Args:
        model (NormalizingFlowModel): Trained model
        points (torch.Tensor): Points to track (shape: (n_points, 2))
        save_path (str, optional): Path to save the plot
    """
    device = next(model.parameters()).device
    
    # Track points through inverse layers
    points_tracked = [points.detach().cpu().numpy()]
    z = points
    for layer in reversed(model.layers):
        z, _ = layer.inverse(z)  # Unpack the tuple, ignore log_det
        points_tracked.append(z.detach().cpu().numpy())
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    points_tracked = np.array(points_tracked)  # Shape: (n_layers + 1, n_points, 2)
    
    # Create a colormap for the layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(points_tracked)))
    
    for i in range(len(points)):
        plt.plot(points_tracked[:, i, 0], points_tracked[:, i, 1], 'o-', alpha=0.5, label=f'Point {i+1}')
    
    # Add colorbar to show layer progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, len(points_tracked)-1))
    plt.colorbar(sm, ax=plt.gca(), label='Layer Index')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Inverse Trajectories through Layers')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_point_probabilities(model, points):
    """
    Compute the log probability of specific points using the model.
    
    Args:
        model (NormalizingFlowModel): Trained model
        points (torch.Tensor): Points to evaluate (shape: (n_points, 2))
        
    Returns:
        tuple: (log_probs, log_pz, log_det, valid_mask)
    """
    device = next(model.parameters()).device
    D = model.D
    
    # Initialize base distribution
    base_distribution = MultivariateNormal(
        loc=torch.zeros(D, device=device),
        covariance_matrix=torch.eye(D, device=device)
    )
    
    # Compute inverse transformation
    z, log_det = model.inverse(points)
    
    # Check for NaNs or Infs
    valid_mask = (
        torch.isfinite(z).all(dim=1) &
        torch.isfinite(log_det)
    )
    
    # Compute log probability in base space only for valid points
    log_pz = torch.full((points.shape[0],), float('nan'), device=device)
    log_probs = torch.full((points.shape[0],), float('nan'), device=device)
    log_det_valid = torch.full((points.shape[0],), float('nan'), device=device)
    if valid_mask.any():
        log_pz_valid = base_distribution.log_prob(z[valid_mask])
        log_probs_valid = log_pz_valid + log_det[valid_mask]
        log_pz[valid_mask] = log_pz_valid
        log_probs[valid_mask] = log_probs_valid
        log_det_valid[valid_mask] = log_det[valid_mask]
    return log_probs, log_pz, log_det_valid, valid_mask 