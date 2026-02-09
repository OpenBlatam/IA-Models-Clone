"""
Activation Factory Module

Factory for creating activation functions.
"""

import logging
import torch.nn as nn

from .gelu import GELU
from .swish import Swish
from .mish import Mish
from .glu import GLU

logger = logging.getLogger(__name__)


class ActivationFactory:
    """Factory for creating activation functions."""
    
    @staticmethod
    def create(activation_type: str, **kwargs) -> nn.Module:
        """
        Create activation function.
        
        Args:
            activation_type: Type of activation.
            **kwargs: Activation-specific arguments.
        
        Returns:
            Activation module.
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
    """Convenience function for creating activations."""
    return ActivationFactory.create(activation_type, **kwargs)



