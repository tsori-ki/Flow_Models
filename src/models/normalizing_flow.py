import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class AffineCouplingLayer(nn.Module):
    def __init__(self, D):
        """
        Initialize the Affine Coupling Layer.
        
        Args:
            D (int): Input dimension (will be split into D/2 for each part)
        """
        super().__init__()
        self.D = D
        self.D_half = D // 2
        
        # Neural network for log(s)
        self.log_s_net = nn.Sequential(
            nn.Linear(self.D_half, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, self.D_half)  # Outputs log(s)
        )
        
        # Neural network for b
        self.b_net = nn.Sequential(
            nn.Linear(self.D_half, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, self.D_half)  # Outputs b
        )

    def forward(self, z):
        """
        Forward pass: z -> y
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (y, log_det_J) where:
                - y is the output tensor of shape (batch_size, D)
                - log_det_J is the log determinant of shape (batch_size,)
        """
        # Split input into two parts
        z_I, z_II = z[:, :self.D_half], z[:, self.D_half:]
        
        # Get log(s) and b from the networks
        log_s = self.log_s_net(z_I)
        b = self.b_net(z_I)
        
        # Compute y_II
        y_II = torch.exp(log_s) * z_II + b
        
        # Compute log determinant (sum of log_s for the transformed part)
        log_det_J = torch.sum(log_s, dim=1)
        
        # Concatenate z_I and y_II
        y = torch.cat([z_I, y_II], dim=1)
        
        return y, log_det_J
    
    def inverse(self, y):
        """
        Inverse pass: y -> z
        
        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (z, log_det_J) where:
                - z is the output tensor of shape (batch_size, D)
                - log_det_J is the log determinant of shape (batch_size,)
        """
        # Split input into two parts
        y_I, y_II = y[:, :self.D_half], y[:, self.D_half:]
        
        # Get log(s) and b from the networks
        log_s = self.log_s_net(y_I)
        b = self.b_net(y_I)
        
        # Compute z_II
        z_II = (y_II - b) / torch.exp(log_s)
        
        # Compute log determinant (negative sum of log_s for the transformed part)
        log_det_J = -torch.sum(log_s, dim=1)
        
        # Concatenate y_I and z_II
        z = torch.cat([y_I, z_II], dim=1)
        
        return z, log_det_J

class PermutationLayer(nn.Module):
    def __init__(self, D):
        """
        Random permutation layer that either swaps dimensions or keeps them the same.
        For 2-D data, this randomly chooses between identity and swap.
        """
        super().__init__()
        self.D = D
        # Randomly choose between identity and swap
        self.should_swap = torch.rand(1) > 0.5
         
    def forward(self, z):
        """
        Forward pass: z -> y
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (y, log_det_J) where:
                - y is the output tensor of shape (batch_size, D)
                - log_det_J is the log determinant (always 0 for permutation)
        """
        y = torch.flip(z, dims=[1]) if self.should_swap else z
        log_det_J = torch.zeros(z.shape[0], device=z.device)
        return y, log_det_J
    
    def inverse(self, y):
        """
        Inverse pass: y -> z
        
        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (z, log_det_J) where:
                - z is the output tensor of shape (batch_size, D)
                - log_det_J is the log determinant (always 0 for permutation)
        """
        z = torch.flip(y, dims=[1]) if self.should_swap else y
        log_det_J = torch.zeros(y.shape[0], device=y.device)
        return z, log_det_J

class NormalizingFlowModel(nn.Module):
    def __init__(self, n_layers=15, D=2):
        """
        Initialize the Normalizing Flow model.
        
        Args:
            n_layers (int): Number of flow layers (affine coupling + permutation)
            D (int): Input dimension
        """
        super().__init__()
        self.D = D
        self.n_layers = n_layers
        
        # Create alternating layers of affine coupling and permutation
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(AffineCouplingLayer(D))
            if i < n_layers - 1:  # Don't add permutation after last coupling
                self.layers.append(PermutationLayer(D))
        
        # Initialize base distribution (standard normal)
        self.base_dist = MultivariateNormal(
            loc=torch.zeros(D),
            covariance_matrix=torch.eye(D)
        )
    
    def forward(self, z):
        """
        Forward pass: z -> x
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (x, log_det_J) where:
                - x is the output tensor of shape (batch_size, D)
                - log_det_J is the total log determinant of shape (batch_size,)
        """
        x = z
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det
            
        return x, log_det_sum
    
    def inverse(self, x):
        """
        Inverse pass: x -> z
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            tuple: (z, log_det_J) where:
                - z is the output tensor of shape (batch_size, D)
                - log_det_J is the total log determinant of shape (batch_size,)
        """
        z = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_sum += log_det
            
        return z, log_det_sum
    
    def log_prob(self, x):
        """
        Compute log probability of x under the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            torch.Tensor: Log probability of shape (batch_size,)
        """
        z, log_det_J = self.inverse(x)
        log_prob_z = self.base_dist.log_prob(z)
        return log_prob_z + log_det_J
    
    def sample(self, n_samples, seed=None):
        """
        Sample from the model.
        
        Args:
            n_samples (int): Number of samples to generate
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            torch.Tensor: Samples of shape (n_samples, D)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Sample from base distribution
        z = self.base_dist.sample((n_samples,))
        
        # Transform through the flow
        x, _ = self.forward(z)
        return x
    
    def get_layer_outputs(self, z):
        """
        Return list of intermediate tensors after each layer is applied (inclusive).
        Useful for visualization / debugging.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, D)
            
        Returns:
            list: List of tensors after each layer
        """
        outputs = []
        x = z
        for layer in self.layers:
            x, _ = layer(x)
            outputs.append(x)
        return outputs 