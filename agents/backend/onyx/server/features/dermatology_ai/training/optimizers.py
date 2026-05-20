"""
Optimizer and scheduler utilities
Factory functions for creating optimizers and schedulers
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer by name
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ("adam", "adamw", "sgd", "rmsprop")
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "sgd":
        momentum = kwargs.pop('momentum', 0.9)
        return optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    num_epochs: int = 100,
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler by name
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler
        num_epochs: Total number of epochs
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler instance
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "cosine" or scheduler_name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            **kwargs
        )
    elif scheduler_name == "step" or scheduler_name == "steplr":
        step_size = kwargs.pop('step_size', 30)
        gamma = kwargs.pop('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == "plateau" or scheduler_name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.pop('factor', 0.5),
            patience=kwargs.pop('patience', 5),
            **kwargs
        )
    elif scheduler_name == "warmup_cosine":
        # Cosine with warmup
        warmup_epochs = kwargs.pop('warmup_epochs', 5)
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs - warmup_epochs
                )
            ],
            milestones=[warmup_epochs]
        )
    elif scheduler_name == "onecycle":
        max_lr = kwargs.pop('max_lr', 1e-3)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_epochs,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any],
    scheduler_config: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create both optimizer and scheduler from configs
    
    Args:
        model: PyTorch model
        optimizer_config: Dictionary with optimizer configuration
        scheduler_config: Dictionary with scheduler configuration (optional)
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = get_optimizer(model, **optimizer_config)
    
    scheduler = None
    if scheduler_config:
        scheduler = get_scheduler(optimizer, **scheduler_config)
    
    return optimizer, scheduler













