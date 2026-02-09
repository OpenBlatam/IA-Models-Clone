"""
Feed-Forward Modules
Feed-forward network components
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class FeedForward(nn.Module):
    """
    Standard feed-forward network
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward network
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (default: 4 * input_dim)
            output_dim: Output dimension (default: input_dim)
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        hidden_dim = hidden_dim or (4 * input_dim)
        output_dim = output_dim or input_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.layers(x)


class ResidualFeedForward(nn.Module):
    """
    Feed-forward with residual connection
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize residual feed-forward
        
        Args:
            dim: Dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        self.ffn = FeedForward(dim, hidden_dim, dim, dropout, activation)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual"""
        return self.norm(x + self.ffn(x))


class GatedFeedForward(nn.Module):
    """
    Gated feed-forward network (GLU variant)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize gated feed-forward
        
        Args:
            dim: Dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        hidden_dim = hidden_dim or (4 * dim)
        
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        gate = torch.sigmoid(self.gate(x))
        up = self.up(x)
        out = self.down(gate * up)
        return self.dropout(out)













