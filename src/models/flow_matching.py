import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

class FlowMatchingModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        """
        Base Flow Matching model.
        
        Args:
            hidden_dim (int): Width of hidden layers
            num_layers (int): Number of hidden layers (4 or 5)
        """
        super().__init__()
        
        # Input dimension is 3 (2D point + time)
        input_dim = 3
        
        # Create MLP layers
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.01))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
        
        # Output layer (2D vector field)
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize base distribution (standard normal)
        self.base_distribution = MultivariateNormal(
            loc=torch.zeros(2),
            covariance_matrix=torch.eye(2)
        )
    
    def forward(self, y, t):
        """
        Forward pass to compute vector field v̂_t.
        
        Args:
            y (torch.Tensor): Points in data space (batch_size, 2)
            t (torch.Tensor): Time points (batch_size,) or scalar
            
        Returns:
            torch.Tensor: Vector field v̂_t (batch_size, 2)
        """
        # Broadcast t if scalar
        if isinstance(t, (int, float)):
            t = torch.full((y.shape[0],), t, device=y.device)
        
        # Concatenate y and t
        x = torch.cat([y, t.unsqueeze(-1)], dim=-1)
        
        # Compute vector field
        v = self.mlp(x)
        return v
    
    def loss(self, batch):
        """
        Compute the Flow Matching loss (Eq. 16).
        
        Args:
            batch (tuple): (y_0, y_1) where:
                y_0: Initial points from base distribution
                y_1: Target points from data distribution
        
        Returns:
            torch.Tensor: Loss value
        """
        y_0, y_1 = batch
        batch_size = y_0.shape[0]
        
        # Sample random time points
        t = torch.rand(batch_size, device=y_0.device)
        
        # Compute interpolated points
        y = t.unsqueeze(-1) * y_1 + (1 - t.unsqueeze(-1)) * y_0
        
        # Compute vector field
        v = self.forward(y, t)
        
        # Compute target vector field (y_1 - y_0)
        v_target = y_1 - y_0
        
        # Compute MSE loss
        loss = F.mse_loss(v, v_target)
        return loss
    
    def sample(self, n_samples, dt=1/1000):
        """
        Generate samples using forward Euler integration.
        
        Args:
            n_samples (int): Number of samples to generate
            dt (float): Time step size
        
        Returns:
            torch.Tensor: Generated samples (n_samples, 2)
        """
        device = next(self.parameters()).device
        
        # Sample initial points from base distribution
        y_0 = self.base_distribution.sample((n_samples,)).to(device)
        
        # Initialize trajectory
        y = y_0
        
        # Euler integration
        t = 0.0
        while t < 1.0:
            v = self.forward(y, t)
            y = y + v * dt
            t += dt
        
        return y
    
    def reverse_sample(self, points, dt=1/1000):
        """
        Reverse flow using backward Euler integration.
        
        Args:
            points (torch.Tensor): Points to transform (batch_size, 2)
            dt (float): Time step size
        
        Returns:
            torch.Tensor: Transformed points (batch_size, 2)
        """
        device = next(self.parameters()).device
        y = points.to(device)
        
        # Euler integration (backward)
        t = 1.0
        while t > 0.0:
            v = self.forward(y, t)
            y = y - v * dt
            t -= dt
        
        return y 

class UnconditionalFlowMatching(FlowMatchingModel):
    """
    Unconditional Flow Matching model for the Olympic rings dataset.
    Inherits all functionality from the base FlowMatchingModel.
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers) 

class ConditionalFlowMatching(FlowMatchingModel):
    """
    Conditional Flow Matching model for the Olympic rings dataset.
    Adds class conditioning to the base FlowMatchingModel.
    """
    def __init__(self, hidden_dim=64, num_layers=4, num_classes=5, emb_dim=32):
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        
        # Additional FC layer to mix class embedding with input
        self.mixing_layer = nn.Linear(3 + emb_dim, hidden_dim)
        
        # Update MLP to start from hidden_dim
        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, y, t, class_id):
        """
        Forward pass to compute vector field v̂_t conditioned on class.
        
        Args:
            y (torch.Tensor): Points in data space (batch_size, 2)
            t (torch.Tensor): Time points (batch_size,) or scalar
            class_id (torch.Tensor): Class indices (batch_size,)
            
        Returns:
            torch.Tensor: Vector field v̂_t (batch_size, 2)
        """
        # Broadcast t if scalar
        if isinstance(t, (int, float)):
            t = torch.full((y.shape[0],), t, device=y.device)
        
        # Get class embeddings
        class_emb = self.class_embedding(class_id)
        
        # Concatenate y, t, and class embedding
        x = torch.cat([y, t.unsqueeze(-1), class_emb], dim=-1)
        
        # Mix through additional FC layer
        x = self.mixing_layer(x)
        x = F.leaky_relu(x, 0.01)
        
        # Compute vector field
        v = self.mlp(x)
        return v
    
    def sample(self, n_samples, class_id, dt=1/1000):
        """
        Generate samples for a specific class using forward Euler integration.
        
        Args:
            n_samples (int): Number of samples to generate
            class_id (int): Class index to generate samples for
            dt (float): Time step size
        
        Returns:
            torch.Tensor: Generated samples (n_samples, 2)
        """
        device = next(self.parameters()).device
        
        # Sample initial points from base distribution
        y_0 = self.base_distribution.sample((n_samples,)).to(device)
        
        # Create class_id tensor
        class_id_tensor = torch.full((n_samples,), class_id, device=device)
        
        # Initialize trajectory
        y = y_0
        
        # Euler integration
        t = 0.0
        while t < 1.0:
            v = self.forward(y, t, class_id_tensor)
            y = y + v * dt
            t += dt
        
        return y
    
    def reverse_sample(self, points, class_id, dt=1/1000):
        """
        Reverse flow using backward Euler integration.
        
        Args:
            points (torch.Tensor): Points to transform (batch_size, 2)
            class_id (torch.Tensor): Class indices (batch_size,)
            dt (float): Time step size
        
        Returns:
            torch.Tensor: Transformed points (batch_size, 2)
        """
        device = next(self.parameters()).device
        y = points.to(device)
        
        # Euler integration (backward)
        t = 1.0
        while t > 0.0:
            v = self.forward(y, t, class_id)
            y = y - v * dt
            t -= dt
        
        return y 

    def loss(self, batch):
        """
        Compute the Flow Matching loss (Eq. 16) for the conditional model.
        Args:
            batch (tuple): (y_0, y_1, class_id) where:
                y_0: Initial points from base distribution
                y_1: Target points from data distribution
                class_id: Class indices (batch_size,)
        Returns:
            torch.Tensor: Loss value
        """
        y_0, y_1, class_id = batch
        batch_size = y_0.shape[0]
        t = torch.rand(batch_size, device=y_0.device)
        y = t.unsqueeze(-1) * y_1 + (1 - t.unsqueeze(-1)) * y_0
        v = self.forward(y, t, class_id)
        v_target = y_1 - y_0
        loss = F.mse_loss(v, v_target)
        return loss 