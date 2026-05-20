"""
Optimizer Factory Module

Factory for creating optimizers.
"""

from typing import Dict, Any
import logging
import torch.optim as optim

from .adam import create_adam
from .adamw import create_adamw
from .sgd import create_sgd
from .rmsprop import create_rmsprop

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(
        optimizer_type: str,
        parameters,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer based on type.
        
        Args:
            optimizer_type: Type of optimizer ("adam", "adamw", "sgd", "rmsprop").
            parameters: Model parameters.
            learning_rate: Learning rate.
            **kwargs: Additional optimizer-specific arguments.
        
        Returns:
            Optimizer instance.
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adam":
            return create_adam(parameters, learning_rate, **kwargs)
        elif optimizer_type == "adamw":
            return create_adamw(parameters, learning_rate, **kwargs)
        elif optimizer_type == "sgd":
            return create_sgd(parameters, learning_rate, **kwargs)
        elif optimizer_type == "rmsprop":
            return create_rmsprop(parameters, learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_optimizer(
    optimizer_type: str,
    parameters,
    learning_rate: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """Convenience function for creating optimizers."""
    return OptimizerFactory.create(optimizer_type, parameters, learning_rate, **kwargs)



