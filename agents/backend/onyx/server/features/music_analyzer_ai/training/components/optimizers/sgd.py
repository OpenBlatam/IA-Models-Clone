"""
SGD Optimizer Module

Creates SGD optimizer instances.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_sgd(
    parameters,
    learning_rate: float = 1e-4,
    **kwargs
) -> optim.SGD:
    """
    Create SGD optimizer.
    
    Args:
        parameters: Model parameters.
        learning_rate: Learning rate.
        **kwargs: Additional arguments (momentum, weight_decay, nesterov).
    
    Returns:
        SGD optimizer instance.
    """
    return optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=kwargs.get("momentum", 0.9),
        weight_decay=kwargs.get("weight_decay", 0.0),
        nesterov=kwargs.get("nesterov", False)
    )



