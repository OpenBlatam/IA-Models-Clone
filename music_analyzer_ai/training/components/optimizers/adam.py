"""
Adam Optimizer Module

Creates Adam optimizer instances.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_adam(
    parameters,
    learning_rate: float = 1e-4,
    **kwargs
) -> optim.Adam:
    """
    Create Adam optimizer.
    
    Args:
        parameters: Model parameters.
        learning_rate: Learning rate.
        **kwargs: Additional arguments (betas, eps, weight_decay, amsgrad).
    
    Returns:
        Adam optimizer instance.
    """
    return optim.Adam(
        parameters,
        lr=learning_rate,
        betas=kwargs.get("betas", (0.9, 0.999)),
        eps=kwargs.get("eps", 1e-8),
        weight_decay=kwargs.get("weight_decay", 0.0),
        amsgrad=kwargs.get("amsgrad", False)
    )



