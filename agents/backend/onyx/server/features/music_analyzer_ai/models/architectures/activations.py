"""
Modular Activation Functions
Separated activation modules for better composability
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class Swish(nn.Module):
    """Swish activation (SiLU)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GLU(nn.Module):
    """Gated Linear Unit"""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class ActivationFactory:
    """Factory for creating activation functions"""
    
    @staticmethod
    def create(activation_type: str, **kwargs) -> nn.Module:
        """
        Create activation function
        
        Args:
            activation_type: Type of activation
            **kwargs: Activation-specific arguments
        
        Returns:
            Activation module
        """
        activation_type = activation_type.lower()
        
        if activation_type == "relu":
            return nn.ReLU(**kwargs)
        elif activation_type == "gelu":
            return GELU()
        elif activation_type == "swish" or activation_type == "silu":
            return Swish()
        elif activation_type == "mish":
            return Mish()
        elif activation_type == "glu":
            return GLU(**kwargs)
        elif activation_type == "tanh":
            return nn.Tanh()
        elif activation_type == "sigmoid":
            return nn.Sigmoid()
        elif activation_type == "elu":
            return nn.ELU(**kwargs)
        elif activation_type == "leaky_relu":
            return nn.LeakyReLU(**kwargs)
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")


def create_activation(activation_type: str, **kwargs) -> nn.Module:
    """Convenience function for creating activations"""
    return ActivationFactory.create(activation_type, **kwargs)



