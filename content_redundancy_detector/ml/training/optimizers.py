"""
Optimizer Factory
Modular optimizer creation and configuration
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """
    Factory for creating optimizers
    """
    
    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_type: str = "sgd",
        learning_rate: float = 0.001,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer instance
        """
        params = model.parameters()
        
        if optimizer_type.lower() == "sgd":
            return optim.SGD(
                params,
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0001),
                nesterov=kwargs.get('nesterov', False),
            )
        elif optimizer_type.lower() == "adam":
            return optim.Adam(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                weight_decay=kwargs.get('weight_decay', 0.0001),
                eps=kwargs.get('eps', 1e-8),
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                weight_decay=kwargs.get('weight_decay', 0.01),
                eps=kwargs.get('eps', 1e-8),
            )
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                params,
                lr=learning_rate,
                alpha=kwargs.get('alpha', 0.99),
                weight_decay=kwargs.get('weight_decay', 0.0001),
                momentum=kwargs.get('momentum', 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_optimizer_from_config(
        model: torch.nn.Module,
        config: Dict[str, Any]
    ) -> optim.Optimizer:
        """
        Create optimizer from configuration dictionary
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
            
        Returns:
            Optimizer instance
        """
        optimizer_type = config.pop('type', 'sgd')
        learning_rate = config.pop('learning_rate', 0.001)
        return OptimizerFactory.create_optimizer(
            model,
            optimizer_type,
            learning_rate,
            **config
        )



