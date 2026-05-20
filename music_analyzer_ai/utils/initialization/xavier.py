"""
Xavier Initialization Module

Implements Xavier/Glorot initialization strategies.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def xavier_uniform(module: nn.Module, gain: float = 1.0):
    """
    Xavier/Glorot initialization.
    
    Args:
        module: PyTorch module to initialize.
        gain: Scaling factor for the weights.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        logger.debug(f"Applied Xavier uniform initialization to {type(module).__name__}")



