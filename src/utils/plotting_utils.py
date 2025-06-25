import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from models.normalizing_flow import AffineCouplingLayer
import os

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
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
                os.makedirs(os.path.dirname(f'{save_dir}/{filename}'), exist_ok=True)
                plt.savefig(f'{save_dir}/{filename}')
            else:
                os.makedirs(os.path.dirname(f'{save_dir}/samples_seed_{seed}.png'), exist_ok=True)
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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
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
        os.makedirs(os.path.dirname(f'{save_dir}/layer_0.png'), exist_ok=True)
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
                    os.makedirs(os.path.dirname(f'{save_dir}/affine_layer_{affine_layer_count}.png'), exist_ok=True)
                    plt.savefig(f'{save_dir}/affine_layer_{affine_layer_count}.png')
                plt.close()

def plot_sampling_trajectories(model, n_points=10, save_path=None):
    """
    Plot the 2D trajectories of points as they pass through the model layers.
    Each segment of the trajectory is colored by its layer index.
    Start and end points of trajectories are emphasized.
    
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
    
    # Generate initial points from the base distribution (latent space z)
    z_initial = base_distribution.sample((n_points,))
    
    layer_outputs = [z_initial.detach().cpu().numpy()] # Start with z0
    current_x = z_initial
    
    print("\n--- Forward Sampling Trajectory Debug ---")
    print(f"Initial z points (Layer 0):\n{layer_outputs[0]}")

    for layer_idx, layer in enumerate(model.layers):
        current_x, _ = layer(current_x)
        layer_outputs.append(current_x.detach().cpu().numpy())
        print(f"Layer {layer_idx + 1} ({type(layer).__name__}) output:\n{layer_outputs[-1]}")
        if not np.all(np.isfinite(layer_outputs[-1])):
            print(f"WARNING: NaN or Inf detected in layer {layer_idx + 1} output. Trajectories might be unstable.")

    print("--- End Forward Sampling Trajectory Debug ---\n")
    
    trajectories_np = np.array(layer_outputs)
    num_layers_plus_one = trajectories_np.shape[0]

    plt.figure(figsize=(10, 10))
    cmap = plt.cm.viridis
    # Normalize color based on layer index for segments (0 to num_layers-1)
    segment_norm = plt.Normalize(vmin=0, vmax=num_layers_plus_one - 2) 

    for point_idx in range(n_points):
        # Plot segments
        for layer_num in range(num_layers_plus_one - 1):
            start_segment_point = trajectories_np[layer_num, point_idx, :]
            end_segment_point = trajectories_np[layer_num + 1, point_idx, :]
            segment_points = np.array([start_segment_point, end_segment_point])
            
            label = f'Point {point_idx+1}' if layer_num == 0 else None # Label only once per trajectory
            plt.plot(segment_points[:, 0], segment_points[:, 1], '-', color=cmap(segment_norm(layer_num)), alpha=0.6, label=label)

        # Emphasize start point (Layer 0)
        plt.plot(trajectories_np[0, point_idx, 0], trajectories_np[0, point_idx, 1], 
                 'o', color=cmap(segment_norm(0)), markersize=8, markeredgecolor='black')
                 # Use color of the first segment for start point, or a distinct color like 'purple'

        # Emphasize end point (Last Layer)
        plt.plot(trajectories_np[-1, point_idx, 0], trajectories_np[-1, point_idx, 1], 
                 'X', color=cmap(segment_norm(num_layers_plus_one - 2)), markersize=8, markeredgecolor='black')
                 # Use color of the last segment for end point, or a distinct color like 'red'
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=segment_norm)
    plt.colorbar(sm, ax=plt.gca(), label='Layer Index (segment start)')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Forward Sampling Trajectories through Layers')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Forward sampling trajectory plot saved to {save_path}")
    plt.close()

def plot_inverse_trajectories(model, points, save_path=None):
    """
    Plot the inverse trajectories of specific points through the model layers.
    Each point's trajectory is plotted with a consistent color and markers at each layer output, connected by lines.
    Handles NaN/Inf values by stopping trajectory tracking for affected points.
    
    Args:
        model (NormalizingFlowModel): Trained model
        points (torch.Tensor): Points to track (shape: (n_points, 2))
        save_path (str, optional): Path to save the plot
    """
    device = next(model.parameters()).device
    points_on_device = points.to(device)
    n_points = points_on_device.shape[0]
    
    trajectories_cpu = [[] for _ in range(n_points)] 
    for i in range(n_points):
        trajectories_cpu[i].append(points_on_device[i].detach().cpu().numpy())

    current_points_gpu = points_on_device.clone()
    active_points_mask = torch.ones(n_points, dtype=torch.bool, device=device)

    # print("\n--- Inverse Trajectory Debug (Q5 Style) ---")
    # print(f"Initial points (data space):\n{current_points_gpu.detach().cpu().numpy()}")

    for layer_idx, layer in enumerate(reversed(model.layers)):
        if not active_points_mask.any():
            break
        
        z_active, _ = layer.inverse(current_points_gpu[active_points_mask])
        current_points_gpu[active_points_mask] = z_active
        is_finite_active = torch.isfinite(z_active).all(dim=1)
        
        active_indices = torch.where(active_points_mask)[0]
        newly_inactive_indices_local = torch.where(~is_finite_active)[0]
        
        if newly_inactive_indices_local.numel() > 0:
            newly_inactive_indices_global = active_indices[newly_inactive_indices_local]
            active_points_mask[newly_inactive_indices_global] = False

        for i_global in range(n_points):
            if active_points_mask[i_global]:
                trajectories_cpu[i_global].append(current_points_gpu[i_global].detach().cpu().numpy())

    # print("--- End Inverse Trajectory Debug ---\n")

    plt.figure(figsize=(6, 6))
    point_colors = plt.cm.get_cmap('tab10', n_points).colors 

    for i in range(n_points):
        if trajectories_cpu[i] and len(trajectories_cpu[i]) > 0: # Check if there are any points to plot
            traj_np = np.array(trajectories_cpu[i]) # Shape: (num_steps, 2)
            plt.plot(traj_np[:, 0], traj_np[:, 1], 'o-', 
                     color=point_colors[i % len(point_colors)], 
                     markersize=5, # Adjusted markersize slightly
                     alpha=0.8, 
                     label=f'Point {i}' if i < 10 else f'P{i}')
            
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Inverse Trajectories (Q5 Style)') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left') 
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Inverse trajectory plot (Q5 style) saved to {save_path}")
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