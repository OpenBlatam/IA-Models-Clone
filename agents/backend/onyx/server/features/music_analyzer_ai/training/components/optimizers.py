"""
Modular Optimizer Factory
Creates optimizers based on configuration
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class OptimizerFactory:
    """Factory for creating optimizers"""
    
    @staticmethod
    def create(
        optimizer_type: str,
        parameters,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer based on type
        
        Args:
            optimizer_type: Type of optimizer ("adam", "adamw", "sgd", etc.)
            parameters: Model parameters
            learning_rate: Learning rate
            **kwargs: Additional optimizer-specific arguments
        
        Returns:
            Optimizer instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for optimizers")
        
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adam":
            return optim.Adam(
                parameters,
                lr=learning_rate,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 0.0),
                amsgrad=kwargs.get("amsgrad", False)
            )
        
        elif optimizer_type == "adamw":
            return optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 1e-5),
                amsgrad=kwargs.get("amsgrad", False)
            )
        
        elif optimizer_type == "sgd":
            return optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=kwargs.get("weight_decay", 0.0),
                nesterov=kwargs.get("nesterov", False)
            )
        
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(
                parameters,
                lr=learning_rate,
                alpha=kwargs.get("alpha", 0.99),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 0.0),
                momentum=kwargs.get("momentum", 0.0)
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_optimizer(
    optimizer_type: str,
    parameters,
    learning_rate: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """Convenience function for creating optimizers"""
    return OptimizerFactory.create(optimizer_type, parameters, learning_rate, **kwargs)



