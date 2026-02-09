"""
RMSprop Optimizer Module

Creates RMSprop optimizer instances.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_rmsprop(
    parameters,
    learning_rate: float = 1e-4,
    **kwargs
) -> optim.RMSprop:
    """
    Create RMSprop optimizer.
    
    Args:
        parameters: Model parameters.
        learning_rate: Learning rate.
        **kwargs: Additional arguments (alpha, eps, weight_decay, momentum).
    
    Returns:
        RMSprop optimizer instance.
    """
    return optim.RMSprop(
        parameters,
        lr=learning_rate,
        alpha=kwargs.get("alpha", 0.99),
        eps=kwargs.get("eps", 1e-8),
        weight_decay=kwargs.get("weight_decay", 0.0),
        momentum=kwargs.get("momentum", 0.0)
    )



