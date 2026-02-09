"""
Optimizers
==========

Utilidades para crear optimizadores.
"""

import logging
from enum import Enum
from typing import Any

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    optim = None

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Tipo de optimizador."""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


def create_optimizer(
    optimizer_type: OptimizerType,
    parameters: Any,
    learning_rate: float = 0.001,
    **kwargs
):
    """
    Crear optimizador.
    
    Args:
        optimizer_type: Tipo de optimizador
        parameters: Parámetros del modelo
        learning_rate: Learning rate
        **kwargs: Argumentos adicionales
        
    Returns:
        Optimizador
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for optimizers")
    
    if optimizer_type == OptimizerType.ADAM:
        return optim.Adam(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == OptimizerType.ADAMW:
        return optim.AdamW(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == OptimizerType.SGD:
        momentum = kwargs.pop("momentum", 0.9)
        return optim.SGD(parameters, lr=learning_rate, momentum=momentum, **kwargs)
    elif optimizer_type == OptimizerType.RMSPROP:
        return optim.RMSprop(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == OptimizerType.ADAGRAD:
        return optim.Adagrad(parameters, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


