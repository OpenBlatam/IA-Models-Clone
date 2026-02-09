"""
GLU (Gated Linear Unit) Module

Implements Gated Linear Unit activation.
"""

import torch
import torch.nn as nn


class GLU(nn.Module):
    """Gated Linear Unit."""
    
    def __init__(self, dim: int = -1):
        """
        Initialize GLU.
        
        Args:
            dim: Dimension along which to split input for gating.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU activation.
        
        Args:
            x: Input tensor (must be even-sized along dim).
        
        Returns:
            Gated output tensor.
        """
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)



