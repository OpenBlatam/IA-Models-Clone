"""
Zeros and Ones Initialization Module

Implements zero and one initialization strategies.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def zeros(module: nn.Module):
    """
    Zero initialization.
    
    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        logger.debug(f"Applied zeros initialization to {type(module).__name__}")


def ones(module: nn.Module):
    """
    Ones initialization.
    
    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        logger.debug(f"Applied ones initialization to {type(module).__name__}")



