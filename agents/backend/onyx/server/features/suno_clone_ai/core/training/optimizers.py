"""
Optimizer Utilities

Provides:
- Optimizer factory functions
- Learning rate scheduler utilities
- Optimizer configuration
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    ExponentialLR,
    LambdaLR
)

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with best practices.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', etc.)
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Beta parameters for Adam/AdamW
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Optimizer instance
    """
    # Get parameters
    if 'parameters' in kwargs:
        parameters = kwargs.pop('parameters')
    else:
        parameters = model.parameters()
    
    optimizer_map = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad
    }
    
    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Available: {list(optimizer_map.keys())}"
        )
    
    optimizer_class = optimizer_map[optimizer_type.lower()]
    
    # Common parameters
    optimizer_params = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
        **kwargs
    }
    
    # Optimizer-specific parameters
    if optimizer_type.lower() in ['adam', 'adamw']:
        optimizer_params['betas'] = betas
        optimizer_params['eps'] = eps
    
    optimizer = optimizer_class(parameters, **optimizer_params)
    
    logger.info(
        f"Created {optimizer_type} optimizer with lr={learning_rate}, "
        f"weight_decay={weight_decay}"
    )
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific parameters
        
    Returns:
        Scheduler instance
    """
    scheduler_map = {
        'cosine': CosineAnnealingLR,
        'step': StepLR,
        'exponential': ExponentialLR,
        'reduce_on_plateau': ReduceLROnPlateau,
        'onecycle': OneCycleLR
    }
    
    if scheduler_type.lower() not in scheduler_map:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: {list(scheduler_map.keys())}"
        )
    
    scheduler_class = scheduler_map[scheduler_type.lower()]
    
    # Default parameters
    default_params = {
        'cosine': {'T_max': 100, 'eta_min': 1e-6},
        'step': {'step_size': 30, 'gamma': 0.1},
        'exponential': {'gamma': 0.95},
        'reduce_on_plateau': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
        'onecycle': {
            'max_lr': kwargs.get('learning_rate', 1e-4),
            'total_steps': kwargs.get('total_steps', 1000)
        }
    }
    
    # Merge default and provided parameters
    params = default_params.get(scheduler_type.lower(), {})
    params.update(kwargs)
    
    scheduler = scheduler_class(optimizer, **params)
    
    logger.info(f"Created {scheduler_type} scheduler")
    
    return scheduler


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    max_lr: float
) -> LambdaLR:
    """
    Create warmup learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        base_lr: Base learning rate
        max_lr: Maximum learning rate
        
    Returns:
        Warmup scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return base_lr + (max_lr - base_lr) * (step / warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max_lr * (0.5 * (1 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get parameter groups with different weight decay.
    
    Useful for applying different weight decay to different layers
    (e.g., no decay for bias and LayerNorm).
    
    Args:
        model: PyTorch model
        weight_decay: Default weight decay
        no_decay_keywords: Keywords for parameters with no decay
        
    Returns:
        List of parameter groups
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    parameter_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    logger.info(
        f"Parameter groups: {len(decay_params)} with decay, "
        f"{len(no_decay_params)} without decay"
    )
    
    return parameter_groups

