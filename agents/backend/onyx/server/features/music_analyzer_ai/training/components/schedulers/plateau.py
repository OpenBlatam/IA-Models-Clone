"""
ReduceLROnPlateau Scheduler Module

Creates reduce on plateau learning rate scheduler.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_plateau_scheduler(
    optimizer: optim.Optimizer,
    **kwargs
) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create plateau scheduler.
    
    Args:
        optimizer: Optimizer instance.
        **kwargs: Additional arguments (mode, factor, patience, verbose).
    
    Returns:
        ReduceLROnPlateau scheduler instance.
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=kwargs.get("mode", "min"),
        factor=kwargs.get("factor", 0.5),
        patience=kwargs.get("patience", 5),
        verbose=kwargs.get("verbose", True)
    )



