"""
Orthogonal Initialization Module

Implements orthogonal initialization strategies.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def orthogonal(module: nn.Module, gain: float = 1.0):
    """
    Orthogonal initialization.
    
    Args:
        module: PyTorch module to initialize.
        gain: Scaling factor for the weights.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        logger.debug(f"Applied orthogonal initialization to {type(module).__name__}")



