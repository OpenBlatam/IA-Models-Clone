"""
Batch Normalization Module

Implements batch normalization for 1D inputs with proper initialization.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D inputs with proper initialization.
    
    Args:
        num_features: Number of features/channels.
        eps: Small value to prevent division by zero.
        momentum: Momentum for running statistics.
        affine: If True, apply learnable affine transformation.
        track_running_stats: If True, track running statistics.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self._reset_parameters()
        logger.debug(f"Initialized BatchNorm1d with num_features={num_features}")
    
    def _reset_parameters(self):
        """Initialize batch norm parameters."""
        if self.bn.affine:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch normalization.
        
        Args:
            x: Input tensor of shape [batch, num_features, ...]
        
        Returns:
            Normalized tensor of same shape
        """
        return self.bn(x)



