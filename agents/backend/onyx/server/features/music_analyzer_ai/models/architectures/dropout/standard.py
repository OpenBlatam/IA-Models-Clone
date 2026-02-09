"""
Standard Dropout Module

Implements standard dropout layer.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StandardDropout(nn.Module):
    """
    Standard dropout layer.
    
    Args:
        p: Dropout probability.
        inplace: If True, do operation in-place.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)
        logger.debug(f"Initialized StandardDropout with p={p}, inplace={inplace}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply standard dropout.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor.
        """
        return self.dropout(x)



