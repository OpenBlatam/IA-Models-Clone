"""
Activation Functions
====================

Funciones de activación avanzadas.
"""

import logging
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    
    Activación GELU.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Swish(nn.Module):
    """
    Swish activation.
    
    x * sigmoid(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """
    Mish activation.
    
    x * tanh(softplus(x))
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * torch.tanh(F.softplus(x))


def get_activation(name: str) -> nn.Module:
    """
    Obtener función de activación por nombre.
    
    Args:
        name: Nombre de activación
        
    Returns:
        Módulo de activación
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "swish": Swish(),
        "mish": Mish(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "elu": nn.ELU(),
        "selu": nn.SELU()
    }
    
    return activations.get(name.lower(), nn.ReLU())

