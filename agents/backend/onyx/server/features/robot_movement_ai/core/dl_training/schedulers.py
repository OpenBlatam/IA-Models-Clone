"""
Learning Rate Schedulers
========================

Utilidades para crear schedulers de learning rate.
"""

import logging
from enum import Enum
from typing import Any

try:
    import torch
    import torch.optim.lr_scheduler as lr_scheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    lr_scheduler = None

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Tipo de scheduler."""
    PLATEAU = "plateau"
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    NONE = "none"


def create_scheduler(
    scheduler_type: SchedulerType,
    optimizer: Any,
    num_epochs: int = 100,
    **kwargs
):
    """
    Crear scheduler.
    
    Args:
        scheduler_type: Tipo de scheduler
        optimizer: Optimizador
        num_epochs: Número de épocas
        **kwargs: Argumentos adicionales
        
    Returns:
        Scheduler o None
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for schedulers")
    
    if scheduler_type == SchedulerType.NONE:
        return None
    elif scheduler_type == SchedulerType.PLATEAU:
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5)
        )
    elif scheduler_type == SchedulerType.COSINE:
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == SchedulerType.STEP:
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        gamma = kwargs.get('gamma', 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


