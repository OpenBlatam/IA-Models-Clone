"""
Post-Norm Residual Connection Module

Implements post-norm residual connection.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PostNormResidual(nn.Module):
    """
    Post-norm residual connection.
    
    Args:
        norm: Normalization module.
        fn: Function to apply.
        dropout: Optional dropout module.
    """
    
    def __init__(
        self,
        norm: nn.Module,
        fn: nn.Module,
        dropout: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.dropout = dropout
        logger.debug("Initialized PostNormResidual")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Post-norm: normalize after function.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor.
        """
        residual = x
        x = self.fn(x)
        if self.dropout:
            x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x



