"""
Specialized Training Factory

Creates training components (optimizers, schedulers, losses, strategies)
following factory pattern with proper configuration.
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from ..interfaces.base import (
    ITrainingStrategy,
    ILossFunction,
    IOptimizer,
    IScheduler
)
from ..training.strategies.enhanced_mixed_precision import EnhancedMixedPrecisionStrategy
from ..training.strategies.standard_strategy import StandardTrainingStrategy
from ..training.components.losses import (
    ClassificationLoss,
    RegressionLoss,
    FocalLoss,
    LabelSmoothingLoss
)
from ..training.components.optimizers import OptimizerFactory, create_optimizer
from ..training.components.schedulers import SchedulerFactory, create_scheduler

logger = logging.getLogger(__name__)


class TrainingFactory:
    """
    Factory for creating training components.
    
    Supports:
    - Optimizers
    - Learning rate schedulers
    - Loss functions
    - Training strategies
    """
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create an optimizer.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimizer instance
        """
        return create_optimizer(
            model,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        **kwargs
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            **kwargs: Additional scheduler arguments
            
        Returns:
            Scheduler instance
        """
        return create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            **kwargs
        )
    
    @staticmethod
    def create_loss(
        loss_type: str = "cross_entropy",
        **kwargs
    ) -> nn.Module:
        """
        Create a loss function.
        
        Args:
            loss_type: Type of loss function
            **kwargs: Additional loss arguments
            
        Returns:
            Loss function instance
        """
        loss_registry = {
            "cross_entropy": ClassificationLoss,
            "mse": RegressionLoss,
            "focal": FocalLoss,
            "label_smoothing": LabelSmoothingLoss,
        }
        
        if loss_type not in loss_registry:
            raise ValueError(
                f"Unknown loss type: {loss_type}. "
                f"Available: {list(loss_registry.keys())}"
            )
        
        loss_class = loss_registry[loss_type]
        return loss_class(**kwargs)
    
    @staticmethod
    def create_strategy(
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        strategy_type: str = "mixed_precision",
        device: str = "cuda",
        **kwargs
    ) -> ITrainingStrategy:
        """
        Create a training strategy.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            strategy_type: Type of strategy
            device: Device to use
            **kwargs: Additional strategy arguments
            
        Returns:
            Training strategy instance
        """
        strategy_registry = {
            "standard": StandardTrainingStrategy,
            "mixed_precision": EnhancedMixedPrecisionStrategy,
        }
        
        if strategy_type not in strategy_registry:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(strategy_registry.keys())}"
            )
        
        strategy_class = strategy_registry[strategy_type]
        return strategy_class(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            **kwargs
        )
    
    @classmethod
    def create_training_setup(
        cls,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create complete training setup from configuration.
        
        Args:
            model: Model to train
            config: Training configuration
            
        Returns:
            Dictionary with optimizer, scheduler, loss, and strategy
        """
        # Create optimizer
        optimizer = cls.create_optimizer(
            model,
            optimizer_type=config.get("optimizer_type", "adam"),
            learning_rate=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            **config.get("optimizer_kwargs", {})
        )
        
        # Create scheduler
        scheduler = None
        if "scheduler" in config:
            scheduler = cls.create_scheduler(
                optimizer,
                scheduler_type=config["scheduler"].get("type", "cosine"),
                **config["scheduler"].get("kwargs", {})
            )
        
        # Create loss
        loss_fn = cls.create_loss(
            loss_type=config.get("loss_type", "cross_entropy"),
            **config.get("loss_kwargs", {})
        )
        
        # Create strategy
        strategy = cls.create_strategy(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            strategy_type=config.get("strategy_type", "mixed_precision"),
            device=config.get("device", "cuda"),
            **config.get("strategy_kwargs", {})
        )
        
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_fn": loss_fn,
            "strategy": strategy
        }


__all__ = [
    "TrainingFactory",
]



