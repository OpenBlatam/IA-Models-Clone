"""
Gated Residual Connection Module

Implements gated residual connection.
"""

from typing import Optional, Callable
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GatedResidual(nn.Module):
    """
    Gated residual connection.
    
    Args:
        gate_fn: Gate function (default: sigmoid).
    """
    
    def __init__(self, gate_fn: Optional[Callable] = None):
        super().__init__()
        self.gate_fn = gate_fn or torch.sigmoid
        logger.debug("Initialized GatedResidual")
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Gated residual connection.
        
        Args:
            x: Input tensor.
            residual: Residual tensor.
        
        Returns:
            Output tensor.
        """
        gate = self.gate_fn(x)
        return gate * x + (1 - gate) * residual



