"""
CNN Model - Convolutional Neural Network Architecture
=====================================================

Implements CNN models following best practices:
- Convolutional layers with proper padding
- Batch normalization
- Dropout
- Residual connections (optional)
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        use_bn: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding (auto-calculated if None)
            use_bn: Use batch normalization
            activation: Activation function ('relu', 'gelu', 'swish')
            dropout: Dropout probability
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Kernel size
            use_bn: Use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = ConvBlock(
            channels, channels, kernel_size,
            use_bn=use_bn, activation='relu', dropout=0.0
        )
        self.conv2 = ConvBlock(
            channels, channels, kernel_size,
            use_bn=use_bn, activation='identity', dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class CNNModel(BaseModel):
    """
    Convolutional Neural Network model.
    
    Configurable architecture with multiple convolutional layers,
    optional residual connections, and adaptive pooling.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        conv_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: Optional[List[int]] = None,
        use_residual: bool = False,
        use_bn: bool = True,
        dropout: float = 0.5,
        activation: str = 'relu',
        pool_type: str = 'adaptive',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CNN model.
        
        Args:
            in_channels: Input channels (3 for RGB, 1 for grayscale)
            num_classes: Number of output classes
            conv_channels: List of channels for each conv layer
            kernel_sizes: List of kernel sizes (defaults to 3 for all)
            use_residual: Use residual connections
            use_bn: Use batch normalization
            dropout: Dropout probability
            activation: Activation function
            pool_type: Pooling type ('adaptive', 'max', 'avg')
            config: Additional configuration
        """
        if config is None:
            config = {}
        config.update({
            'in_channels': in_channels,
            'num_classes': num_classes,
            'conv_channels': conv_channels,
            'use_residual': use_residual,
            'use_bn': use_bn,
            'dropout': dropout,
        })
        super().__init__(config)
        
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_channels)
        
        # Build convolutional layers
        layers = []
        prev_channels = in_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            if use_residual and i > 0 and prev_channels == out_channels:
                layers.append(ResidualBlock(out_channels, kernel_size, use_bn, dropout))
            else:
                layers.append(ConvBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=0.0
                ))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            prev_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global pooling
        if pool_type == 'adaptive':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



