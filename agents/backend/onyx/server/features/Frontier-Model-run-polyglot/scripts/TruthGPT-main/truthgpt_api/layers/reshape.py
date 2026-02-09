"""
Reshape Layers for TruthGPT API
===============================

TensorFlow-like reshape layer implementations.
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Optional


class Flatten(nn.Module):
    """
    Flatten layer.
    
    Similar to tf.keras.layers.Flatten, this layer flattens
    the input without affecting the batch size.
    """
    
    def __init__(self, 
                 data_format: str = 'channels_last',
                 name: Optional[str] = None):
        """
        Initialize Flatten layer.
        
        Args:
            data_format: Data format ('channels_last' or 'channels_first')
            name: Optional name for the layer
        """
        super().__init__()
        
        self.data_format = data_format
        self.name = name or "flatten"
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Flattened tensor
        """
        # Flatten all dimensions except batch dimension
        return inputs.view(inputs.size(0), -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"Flatten(data_format={self.data_format})"


class Reshape(nn.Module):
    """
    Reshape layer.
    
    Similar to tf.keras.layers.Reshape, this layer reshapes
    the input to the specified target shape.
    """
    
    def __init__(self, 
                 target_shape: Union[int, Tuple[int, ...]],
                 name: Optional[str] = None):
        """
        Initialize Reshape layer.
        
        Args:
            target_shape: Target shape (excluding batch dimension)
            name: Optional name for the layer
        """
        super().__init__()
        
        self.target_shape = target_shape if isinstance(target_shape, tuple) else (target_shape,)
        self.name = name or f"reshape_{self.target_shape}"
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Reshaped tensor
        """
        # Reshape to target shape
        return inputs.view(inputs.size(0), *self.target_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"Reshape(target_shape={self.target_shape})"









