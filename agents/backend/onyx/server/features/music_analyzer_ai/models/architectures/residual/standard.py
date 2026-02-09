"""
Standard Residual Connection Module

Implements standard residual connection.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualConnection(nn.Module):
    """
    Standard residual connection.
    
    Args:
        dropout: Optional dropout module.
    """
    
    def __init__(self, dropout: Optional[nn.Module] = None):
        super().__init__()
        self.dropout = dropout
        logger.debug("Initialized ResidualConnection")
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Add residual connection.
        
        Args:
            x: Input tensor.
            residual: Residual tensor.
        
        Returns:
            Output tensor.
        """
        if self.dropout:
            x = self.dropout(x)
        return x + residual



