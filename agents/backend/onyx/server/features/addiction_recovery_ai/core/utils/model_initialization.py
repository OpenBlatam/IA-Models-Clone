"""
Model Initialization Utilities
Better weight initialization strategies
"""

import torch
import torch.nn as nn
import math
from typing import Optional

logger = logging.getLogger(__name__)
import logging


def init_weights_xavier(module: nn.Module):
    """
    Xavier/Glorot initialization
    
    Args:
        module: Module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weights_kaiming(module: nn.Module):
    """
    Kaiming/He initialization
    
    Args:
        module: Module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weights_orthogonal(module: nn.Module):
    """
    Orthogonal initialization
    
    Args:
        module: Module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weights_normal(module: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Normal initialization
    
    Args:
        module: Module to initialize
        mean: Mean
        std: Standard deviation
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weights_zero(module: nn.Module):
    """
    Zero initialization
    
    Args:
        module: Module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def initialize_model(
    model: nn.Module,
    method: str = "xavier",
    **kwargs
):
    """
    Initialize model weights
    
    Args:
        model: Model to initialize
        method: Initialization method (xavier, kaiming, orthogonal, normal, zero)
        **kwargs: Additional arguments
    """
    methods = {
        "xavier": init_weights_xavier,
        "kaiming": init_weights_kaiming,
        "orthogonal": init_weights_orthogonal,
        "normal": lambda m: init_weights_normal(m, **kwargs),
        "zero": init_weights_zero
    }
    
    if method not in methods:
        raise ValueError(f"Unknown initialization method: {method}")
    
    model.apply(methods[method])
    logger.info(f"Model initialized with {method} method")









