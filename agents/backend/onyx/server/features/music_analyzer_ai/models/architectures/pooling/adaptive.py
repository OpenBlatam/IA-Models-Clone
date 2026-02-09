"""
Adaptive Pooling Module

Implements adaptive pooling that can switch between strategies.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

from .mean import MeanPooling
from .max import MaxPooling
from .attention import AttentionPooling

logger = logging.getLogger(__name__)


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that can switch between strategies.
    
    Args:
        strategy: Pooling strategy ("mean", "max", "attention").
        embed_dim: Embedding dimension (required for attention pooling).
    """
    
    def __init__(self, strategy: str = "mean", embed_dim: Optional[int] = None):
        super().__init__()
        self.strategy = strategy
        
        if strategy == "mean":
            self.pooler = MeanPooling()
        elif strategy == "max":
            self.pooler = MaxPooling()
        elif strategy == "attention":
            if embed_dim is None:
                raise ValueError("embed_dim required for attention pooling")
            self.pooler = AttentionPooling(embed_dim)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
        
        logger.debug(f"Initialized AdaptivePooling with strategy='{strategy}'")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            mask: Optional mask.
        
        Returns:
            Pooled tensor.
        """
        return self.pooler(x, mask)



