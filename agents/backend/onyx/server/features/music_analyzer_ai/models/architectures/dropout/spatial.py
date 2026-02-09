"""
Spatial Dropout Module

Implements spatial dropout for 2D/3D tensors.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpatialDropout(nn.Module):
    """
    Spatial dropout for 2D/3D tensors.
    
    Args:
        p: Dropout probability.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        logger.debug(f"Initialized SpatialDropout with p={p}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial dropout.
        
        Args:
            x: Input tensor [batch, seq_len, features] or [batch, channels, height, width].
        
        Returns:
            Output tensor.
        """
        if x.dim() == 3:  # [batch, seq_len, features]
            # Drop entire feature vectors
            mask = torch.bernoulli(torch.ones(x.shape[:2]) * (1 - self.p))
            mask = mask.unsqueeze(-1).to(x.device)
            return x * mask / (1 - self.p)
        elif x.dim() == 4:  # [batch, channels, height, width]
            # Drop entire channels
            mask = torch.bernoulli(torch.ones(x.shape[:2]) * (1 - self.p))
            mask = mask.unsqueeze(-1).unsqueeze(-1).to(x.device)
            return x * mask / (1 - self.p)
        else:
            # Fallback to standard dropout
            return nn.functional.dropout(x, p=self.p, training=self.training)



