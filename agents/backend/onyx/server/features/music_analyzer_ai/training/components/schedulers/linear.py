"""
Linear Scheduler Module

Creates linear learning rate scheduler.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_linear_scheduler(
    optimizer: optim.Optimizer,
    **kwargs
) -> optim.lr_scheduler.LinearLR:
    """
    Create linear scheduler.
    
    Args:
        optimizer: Optimizer instance.
        **kwargs: Additional arguments (start_factor, end_factor, total_iters).
    
    Returns:
        LinearLR scheduler instance.
    """
    return optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=kwargs.get("start_factor", 1.0),
        end_factor=kwargs.get("end_factor", 0.1),
        total_iters=kwargs.get("total_iters", 100)
    )



