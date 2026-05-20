"""
Modular Weight Initialization Utilities
Centralized initialization strategies
"""

from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class WeightInitializer:
    """
    Centralized weight initialization strategies
    """
    
    @staticmethod
    def kaiming_uniform(
        module: nn.Module,
        a: float = math.sqrt(5),
        mode: str = 'fan_in',
        nonlinearity: str = 'relu'
    ):
        """Kaiming/He initialization for ReLU activations"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
    
    @staticmethod
    def xavier_uniform(module: nn.Module, gain: float = 1.0):
        """Xavier/Glorot initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def orthogonal(module: nn.Module, gain: float = 1.0):
        """Orthogonal initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def normal(module: nn.Module, mean: float = 0.0, std: float = 0.02):
        """Normal initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def zeros(module: nn.Module):
        """Zero initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def ones(module: nn.Module):
        """Ones initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def lstm_weights(module: nn.LSTM):
        """Proper LSTM weight initialization"""
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
    
    @staticmethod
    def transformer_weights(module: nn.Module):
        """Transformer-specific initialization"""
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                # Use Xavier for most layers
                nn.init.xavier_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, nn.Embedding):
                nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
            elif isinstance(submodule, nn.LayerNorm):
                nn.init.ones_(submodule.weight)
                nn.init.zeros_(submodule.bias)


def initialize_weights(
    module: nn.Module,
    strategy: str = "xavier",
    **kwargs
):
    """
    Initialize weights using specified strategy
    
    Args:
        module: Module to initialize
        strategy: Initialization strategy
        **kwargs: Strategy-specific arguments
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



