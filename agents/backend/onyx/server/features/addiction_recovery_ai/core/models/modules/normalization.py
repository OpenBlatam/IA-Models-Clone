"""
Normalization Modules
Various normalization layers
"""

import torch
import torch.nn as nn
from typing import Optional


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Initialize layer norm
        
        Args:
            normalized_shape: Shape to normalize
            eps: Epsilon for numerical stability
            elementwise_affine: Use learnable affine parameters
        """
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.norm(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Initialize RMS norm
        
        Args:
            dim: Dimension
            eps: Epsilon
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive layer normalization
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize adaptive layer norm
        
        Args:
            dim: Dimension
            eps: Epsilon
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.adaptive_weight = nn.Linear(dim, dim)
        self.adaptive_bias = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            condition: Conditioning tensor (optional)
            
        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        
        if condition is not None:
            adaptive_w = self.adaptive_weight(condition)
            adaptive_b = self.adaptive_bias(condition)
            return adaptive_w * x_norm + adaptive_b
        
        return self.weight * x_norm + self.bias













