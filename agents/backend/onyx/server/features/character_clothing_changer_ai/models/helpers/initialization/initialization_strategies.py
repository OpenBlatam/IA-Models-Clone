"""
Initialization Strategies
==========================

Different weight initialization strategies for neural networks.
"""

import torch
import torch.nn as nn
from typing import Optional


class InitializationStrategies:
    """Collection of initialization strategies."""
    
    @staticmethod
    def xavier_uniform(layer: nn.Module, gain: float = 1.0) -> None:
        """Xavier uniform initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def xavier_normal(layer: nn.Module, gain: float = 1.0) -> None:
        """Xavier normal initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_uniform(layer: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """Kaiming uniform initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_normal(layer: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """Kaiming normal initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def orthogonal(layer: nn.Module, gain: float = 1.0) -> None:
        """Orthogonal initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def zeros(layer: nn.Module) -> None:
        """Zero initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.zeros_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def ones(layer: nn.Module) -> None:
        """Ones initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.ones_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def normal(layer: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
        """Normal distribution initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(layer.weight, mean=mean, std=std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def uniform(layer: nn.Module, a: float = 0.0, b: float = 1.0) -> None:
        """Uniform distribution initialization."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.uniform_(layer.weight, a=a, b=b)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)


