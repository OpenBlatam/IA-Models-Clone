"""
Alpha Dropout Module

Implements alpha dropout for self-normalizing networks.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AlphaDropout(nn.Module):
    """
    Alpha dropout for self-normalizing networks.
    
    Args:
        p: Dropout probability.
        alpha: Alpha parameter (default: -1.7580993408473766).
    """
    
    def __init__(self, p: float = 0.5, alpha: float = -1.7580993408473766):
        super().__init__()
        self.p = p
        self.alpha = alpha
        logger.debug(f"Initialized AlphaDropout with p={p}, alpha={alpha}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply alpha dropout.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor.
        """
        if not self.training:
            return x
        
        # Alpha dropout formula
        alpha = self.alpha
        keep_prob = 1 - self.p
        noise = torch.rand_like(x)
        noise = noise + alpha
        noise = torch.bernoulli(noise)
        noise = noise * (keep_prob + alpha * keep_prob)
        
        return x * noise



