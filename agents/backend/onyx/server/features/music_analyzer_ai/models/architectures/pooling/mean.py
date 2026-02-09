"""
Mean Pooling Module

Implements mean pooling over sequence dimension.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MeanPooling(nn.Module):
    """
    Mean pooling over sequence dimension.
    
    Args:
        dim: Dimension to pool over (default: 1).
    """
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        logger.debug(f"Initialized MeanPooling with dim={dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply mean pooling.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            Pooled tensor [batch, features]
        """
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            masked_sum = (x * mask_expanded).sum(dim=self.dim)
            mask_count = mask_expanded.sum(dim=self.dim)
            return masked_sum / (mask_count + 1e-8)
        return x.mean(dim=self.dim)



