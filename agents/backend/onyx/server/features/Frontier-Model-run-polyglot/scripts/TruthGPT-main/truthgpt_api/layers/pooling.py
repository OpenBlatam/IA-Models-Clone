"""
Pooling Layers for TruthGPT API
===============================

TensorFlow-like pooling layer implementations.
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Optional


class MaxPooling2D(nn.Module):
    """
    2D Max pooling layer.
    
    Similar to tf.keras.layers.MaxPool2D, this layer applies
    max pooling to reduce spatial dimensions.
    """
    
    def __init__(self, 
                 pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Union[int, Tuple[int, int]] = None,
                 padding: Union[str, int, Tuple[int, int]] = 'valid',
                 name: Optional[str] = None):
        """
        Initialize MaxPooling2D layer.
        
        Args:
            pool_size: Size of the pooling window
            strides: Stride of the pooling operation
            padding: Padding mode
            name: Optional name for the layer
        """
        super().__init__()
        
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        self.padding = padding
        self.name = name or f"max_pooling2d_{self.pool_size}"
        
        # Create PyTorch max pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=self.pool_size,
            stride=self.strides,
            padding=self._get_padding()
        )
    
    def _get_padding(self):
        """Get padding value for PyTorch."""
        if self.padding == 'valid':
            return 0
        elif self.padding == 'same':
            return 'same'  # PyTorch handles this automatically
        elif isinstance(self.padding, int):
            return self.padding
        elif isinstance(self.padding, tuple):
            return self.padding
        else:
            return 0
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        return self.pool(inputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"MaxPooling2D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})"


class AveragePooling2D(nn.Module):
    """
    2D Average pooling layer.
    
    Similar to tf.keras.layers.AveragePooling2D, this layer applies
    average pooling to reduce spatial dimensions.
    """
    
    def __init__(self, 
                 pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Union[int, Tuple[int, int]] = None,
                 padding: Union[str, int, Tuple[int, int]] = 'valid',
                 name: Optional[str] = None):
        """
        Initialize AveragePooling2D layer.
        
        Args:
            pool_size: Size of the pooling window
            strides: Stride of the pooling operation
            padding: Padding mode
            name: Optional name for the layer
        """
        super().__init__()
        
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        self.padding = padding
        self.name = name or f"average_pooling2d_{self.pool_size}"
        
        # Create PyTorch average pooling layer
        self.pool = nn.AvgPool2d(
            kernel_size=self.pool_size,
            stride=self.strides,
            padding=self._get_padding()
        )
    
    def _get_padding(self):
        """Get padding value for PyTorch."""
        if self.padding == 'valid':
            return 0
        elif self.padding == 'same':
            return 'same'  # PyTorch handles this automatically
        elif isinstance(self.padding, int):
            return self.padding
        elif isinstance(self.padding, tuple):
            return self.padding
        else:
            return 0
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        return self.pool(inputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"AveragePooling2D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})"









