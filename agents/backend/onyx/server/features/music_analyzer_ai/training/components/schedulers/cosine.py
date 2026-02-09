"""
Cosine Annealing Scheduler Module

Creates cosine annealing learning rate scheduler.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_cosine_scheduler(
    optimizer: optim.Optimizer,
    **kwargs
) -> optim.lr_scheduler.CosineAnnealingLR:
    """
    Create cosine annealing scheduler.
    
    Args:
        optimizer: Optimizer instance.
        **kwargs: Additional arguments (T_max, eta_min).
    
    Returns:
        CosineAnnealingLR scheduler instance.
    """
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=kwargs.get("T_max", 100),
        eta_min=kwargs.get("eta_min", 1e-6)
    )



