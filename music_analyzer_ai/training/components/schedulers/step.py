"""
Step Scheduler Module

Creates step learning rate scheduler.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_step_scheduler(
    optimizer: optim.Optimizer,
    **kwargs
) -> optim.lr_scheduler.StepLR:
    """
    Create step scheduler.
    
    Args:
        optimizer: Optimizer instance.
        **kwargs: Additional arguments (step_size, gamma).
    
    Returns:
        StepLR scheduler instance.
    """
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=kwargs.get("step_size", 30),
        gamma=kwargs.get("gamma", 0.1)
    )



