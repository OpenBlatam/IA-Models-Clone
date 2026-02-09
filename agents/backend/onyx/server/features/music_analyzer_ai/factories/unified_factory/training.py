"""
Training Factory Module

Training component creation (optimizer, scheduler, training loop, callbacks).
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...training.components import create_optimizer, create_scheduler
from ...training.components.callbacks import (
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricsCallback
)
from ...training.loops import StandardTrainingLoop


class TrainingFactoryMixin:
    """Training factory mixin."""
    
    def create_optimizer(
        self,
        optimizer_type: str,
        parameters,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create optimizer.
        
        Args:
            optimizer_type: Optimizer type
            parameters: Model parameters
            config: Optimizer configuration
        
        Returns:
            Optimizer instance
        """
        config = config or {}
        learning_rate = config.pop("learning_rate", 1e-4)
        
        # Try registry first
        optimizer_factory = self.registry.get_optimizer_factory(optimizer_type)
        if optimizer_factory:
            return optimizer_factory(parameters, learning_rate, **config)
        
        # Fallback to standard factory
        return create_optimizer(optimizer_type, parameters, learning_rate, **config)
    
    def create_scheduler(
        self,
        scheduler_type: str,
        optimizer,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create learning rate scheduler.
        
        Args:
            scheduler_type: Scheduler type
            optimizer: Optimizer instance
            config: Scheduler configuration
        
        Returns:
            Scheduler instance
        """
        config = config or {}
        
        # Try registry first
        scheduler_factory = self.registry.get_scheduler_factory(scheduler_type)
        if scheduler_factory:
            return scheduler_factory(optimizer, **config)
        
        # Fallback to standard factory
        return create_scheduler(scheduler_type, optimizer, **config)
    
    def create_training_loop(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ) -> StandardTrainingLoop:
        """
        Create training loop.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            loss_fn: Loss function
            config: Training loop configuration
        
        Returns:
            Training loop instance
        """
        config = config or {}
        
        return StandardTrainingLoop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=config.get("device", "cuda"),
            use_mixed_precision=config.get("use_mixed_precision", True),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            max_grad_norm=config.get("max_grad_norm", 1.0)
        )
    
    def create_callbacks(
        self,
        callback_configs: Dict[str, Dict[str, Any]]
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            callback_configs: Dictionary of callback configurations
        
        Returns:
            List of callback instances
        """
        callbacks = []
        
        for callback_type, config in callback_configs.items():
            # Try registry first
            callback_class = self.registry.get_callback(callback_type)
            if callback_class:
                callbacks.append(callback_class(**config))
                continue
            
            # Fallback to built-in callbacks
            if callback_type == "early_stopping":
                callbacks.append(EarlyStoppingCallback(**config))
            elif callback_type == "checkpoint":
                callbacks.append(CheckpointCallback(**config))
            elif callback_type == "metrics":
                callbacks.append(MetricsCallback(**config))
            else:
                logger.warning(f"Unknown callback type: {callback_type}")
        
        return callbacks



