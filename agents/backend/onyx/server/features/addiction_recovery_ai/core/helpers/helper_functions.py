"""
Helper Functions
Common utility functions
"""

import torch
import torch.nn as nn
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get best available device
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters
    
    Args:
        model: Model
        trainable_only: Count only trainable parameters
        
    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers
    
    Args:
        model: Model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            logger.debug(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
    """
    Unfreeze layers
    
    Args:
        model: Model
        layer_names: List of layer names to unfreeze (None for all)
    """
    for name, param in model.named_parameters():
        if layer_names is None or any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            logger.debug(f"Unfrozen layer: {name}")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer
    
    Args:
        optimizer: Optimizer
        lr: Learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info(f"Learning rate set to {lr}")


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate
    
    Args:
        optimizer: Optimizer
        
    Returns:
        Learning rate
    """
    return optimizer.param_groups[0]['lr']


def format_number(num: float, precision: int = 4) -> str:
    """
    Format number for display
    
    Args:
        num: Number
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if abs(num) < 1e-6:
        return f"{num:.{precision}e}"
    elif abs(num) >= 1e6:
        return f"{num:.{precision}e}"
    else:
        return f"{num:.{precision}f}"








