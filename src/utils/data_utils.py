import torch
import numpy as np
from utils.create_data import create_olympic_rings, create_unconditional_olympic_rings

def get_unconditional_data(n_points=250000, ring_thickness=0.25, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Get unconditional Olympic rings data and convert to PyTorch tensor.
    
    Args:
        n_points (int): Number of points to generate
        ring_thickness (float): Thickness of the rings
        device (str): Device to place the tensor on
        
    Returns:
        torch.Tensor: Data points of shape (n_points, 2)
    """
    data = create_unconditional_olympic_rings(n_points, ring_thickness, verbose=False)
    return torch.tensor(data, dtype=torch.float32, device=device)

def get_conditional_data(n_points=250000, ring_thickness=0.25, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Get conditional Olympic rings data with class labels and convert to PyTorch tensors.
    
    Args:
        n_points (int): Number of points to generate
        ring_thickness (float): Thickness of the rings
        device (str): Device to place the tensors on
        
    Returns:
        tuple: (data_points, labels, label_mapping)
            - data_points: torch.Tensor of shape (n_points, 2)
            - labels: torch.Tensor of shape (n_points,)
            - label_mapping: dict mapping integer labels to color names
    """
    data, labels, label_mapping = create_olympic_rings(n_points, ring_thickness, verbose=False)
    return (torch.tensor(data, dtype=torch.float32, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
            label_mapping)

def create_data_loader(data, batch_size=128, shuffle=True):
    """
    Create a PyTorch DataLoader for the given data.
    
    Args:
        data (torch.Tensor): Data tensor of shape (n_points, n_features)
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the data
    """
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 