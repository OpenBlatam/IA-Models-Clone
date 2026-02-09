"""
Activation Functions

Implements various activation functions for deep learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GELU(nn.Module):
    """Gaussian Error Linear Unit."""
    
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


def create_activation(activation: str) -> nn.Module:
    """
    Create activation function from string.
    
    Args:
        activation: Activation name ('relu', 'gelu', 'swish', 'glu', etc.)
        
    Returns:
        Activation module
    """
    activation_map = {
        'relu': nn.ReLU(),
        'gelu': GELU(),
        'swish': Swish(),
        'glu': GLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'softmax': nn.Softmax(dim=-1),
        'softplus': nn.Softplus()
    }
    
    if activation.lower() not in activation_map:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Available: {list(activation_map.keys())}"
        )
    
    return activation_map[activation.lower()]



