"""
Layer Normalization Implementation

Layer normalization with proper initialization.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    
    Normalizes across the last dimension.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.
        
        Args:
            d_model: Model dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



