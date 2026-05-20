"""
Normal Initialization Module

Implements normal/Gaussian initialization strategies.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def normal(module: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Normal/Gaussian initialization.
    
    Args:
        module: PyTorch module to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        logger.debug(f"Applied normal initialization to {type(module).__name__}")



