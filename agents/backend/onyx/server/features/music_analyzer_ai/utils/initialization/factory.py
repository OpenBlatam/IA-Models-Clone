"""
Initialization Factory Module

Factory for applying initialization strategies to modules.
"""

from typing import Optional
import logging
import torch.nn as nn

from .kaiming import kaiming_uniform
from .xavier import xavier_uniform
from .orthogonal import orthogonal
from .normal import normal
from .zeros_ones import zeros, ones
from .specialized import lstm_weights, transformer_weights

logger = logging.getLogger(__name__)


class WeightInitializer:
    """
    Centralized weight initialization strategies.
    """
    
    @staticmethod
    def kaiming_uniform(
        module: nn.Module,
        a: float = 1.4142135623730951,  # sqrt(2)
        mode: str = 'fan_in',
        nonlinearity: str = 'relu'
    ):
        """Kaiming/He initialization for ReLU activations."""
        kaiming_uniform(module, a=a, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def xavier_uniform(module: nn.Module, gain: float = 1.0):
        """Xavier/Glorot initialization."""
        xavier_uniform(module, gain=gain)
    
    @staticmethod
    def orthogonal(module: nn.Module, gain: float = 1.0):
        """Orthogonal initialization."""
        orthogonal(module, gain=gain)
    
    @staticmethod
    def normal(module: nn.Module, mean: float = 0.0, std: float = 0.02):
        """Normal initialization."""
        normal(module, mean=mean, std=std)
    
    @staticmethod
    def zeros(module: nn.Module):
        """Zero initialization."""
        zeros(module)
    
    @staticmethod
    def ones(module: nn.Module):
        """Ones initialization."""
        ones(module)
    
    @staticmethod
    def lstm_weights(module: nn.LSTM):
        """Proper LSTM weight initialization."""
        lstm_weights(module)
    
    @staticmethod
    def transformer_weights(module: nn.Module):
        """Transformer-specific initialization."""
        transformer_weights(module)


def initialize_weights(
    module: nn.Module,
    strategy: str = "xavier",
    **kwargs
):
    """
    Initialize weights using specified strategy.
    
    Args:
        module: Module to initialize.
        strategy: Initialization strategy.
        **kwargs: Strategy-specific arguments.
    """
    initializer = WeightInitializer()
    
    if strategy == "kaiming":
        for m in module.modules():
            initializer.kaiming_uniform(m, **kwargs)
    elif strategy == "xavier":
        for m in module.modules():
            initializer.xavier_uniform(m, **kwargs)
    elif strategy == "orthogonal":
        for m in module.modules():
            initializer.orthogonal(m, **kwargs)
    elif strategy == "normal":
        for m in module.modules():
            initializer.normal(m, **kwargs)
    elif strategy == "transformer":
        initializer.transformer_weights(module)
    elif strategy == "lstm":
        for m in module.modules():
            if isinstance(m, nn.LSTM):
                initializer.lstm_weights(m)
    else:
        logger.warning(f"Unknown initialization strategy: {strategy}")



