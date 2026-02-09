"""
Pre-Norm Residual Connection Module

Implements pre-norm residual connection.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PreNormResidual(nn.Module):
    """
    Pre-norm residual connection.
    
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
        logger.debug("Initialized PreNormResidual")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-norm: normalize before function.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor.
        """
        residual = x
        x = self.norm(x)
        x = self.fn(x)
        if self.dropout:
            x = self.dropout(x)
        return x + residual



