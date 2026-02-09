"""
Regularization Layers

Implements dropout, drop path, and stochastic depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random


class Dropout(nn.Module):
    """Standard dropout."""
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.dropout = nn.Dropout(p, inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(
            (x.shape[0],) + (1,) * (x.ndim - 1),
            device=x.device
        )
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class StochasticDepth(nn.Module):
    """Stochastic depth for residual connections."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_path = DropPath(drop_prob)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to residual connection.
        
        Args:
            x: Main path output
            residual: Residual connection
            
        Returns:
            x + drop_path(residual)
        """
        return x + self.drop_path(residual)



