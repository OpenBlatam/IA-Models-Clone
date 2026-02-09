"""
Max Pooling Module

Implements max pooling over sequence dimension.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MaxPooling(nn.Module):
    """
    Max pooling over sequence dimension.
    
    Args:
        dim: Dimension to pool over (default: 1).
    """
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        logger.debug(f"Initialized MaxPooling with dim={dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply max pooling with optional mask.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            Pooled tensor [batch, features]
        """
        if mask is not None:
            # Set masked positions to very negative values
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = x * mask_expanded + (1 - mask_expanded) * float('-inf')
        return x.max(dim=self.dim)[0]



