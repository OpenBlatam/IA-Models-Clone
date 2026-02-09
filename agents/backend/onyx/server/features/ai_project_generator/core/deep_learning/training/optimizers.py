"""
Optimizer and Scheduler Utilities
==================================

Factory functions for creating optimizers and schedulers following best practices.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer with best practices.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam/AdamW
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Configured optimizer
    """
    params = model.parameters()
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type == "sgd":
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **{k: v for k, v in kwargs.items() if k not in ['momentum', 'nesterov']}
        )
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 10,
    warmup_epochs: int = 0,
    **kwargs
) -> Optional[Any]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'linear', 'exponential')
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional scheduler-specific parameters
        
    Returns:
        Configured scheduler or None
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        T_max = kwargs.get('T_max', num_epochs)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_type == "step":
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == "plateau":
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 5)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    elif scheduler_type == "linear":
        total_steps = kwargs.get('total_steps', num_epochs)
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.0),
            total_iters=total_steps
        )
    elif scheduler_type == "exponential":
        gamma = kwargs.get('gamma', 0.95)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "none" or scheduler_type is None:
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler



