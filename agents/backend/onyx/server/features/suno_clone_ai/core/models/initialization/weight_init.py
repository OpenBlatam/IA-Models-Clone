"""
Weight Initialization Strategies

Implements best practices for weight initialization.
"""

import torch
import torch.nn as nn
import math


def initialize_linear(module: nn.Linear) -> None:
    """
    Initialize linear layer weights.
    
    Uses Xavier/Glorot uniform initialization.
    
    Args:
        module: Linear layer to initialize
    """
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def initialize_conv(module: nn.Module) -> None:
    """
    Initialize convolutional layer weights.
    
    Uses Kaiming initialization for ReLU, Xavier for others.
    
    Args:
        module: Convolutional layer to initialize
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def initialize_embedding(module: nn.Embedding) -> None:
    """
    Initialize embedding layer weights.
    
    Uses normal distribution initialization.
    
    Args:
        module: Embedding layer to initialize
    """
    nn.init.normal_(module.weight, mean=0.0, std=0.02)


def initialize_layer_norm(module: nn.Module) -> None:
    """
    Initialize layer normalization.
    
    Args:
        module: LayerNorm module to initialize
    """
    if isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize model weights using best practices.
    
    Applies appropriate initialization based on layer type.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        initialize_linear(module)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        initialize_conv(module)
    elif isinstance(module, nn.Embedding):
        initialize_embedding(module)
    elif isinstance(module, nn.LayerNorm):
        initialize_layer_norm(module)



