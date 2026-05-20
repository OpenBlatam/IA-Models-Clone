"""
Batch Normalization Layer for TruthGPT API
==========================================

TensorFlow-like BatchNormalization layer implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple


class BatchNormalization(nn.Module):
    """
    Batch normalization layer.
    
    Similar to tf.keras.layers.BatchNormalization, this layer
    normalizes the inputs to have zero mean and unit variance.
    """
    
    def __init__(self, 
                 axis: int = -1,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 center: bool = True,
                 scale: bool = True,
                 beta_initializer: str = 'zeros',
                 gamma_initializer: str = 'ones',
                 moving_mean_initializer: str = 'zeros',
                 moving_variance_initializer: str = 'ones',
                 name: Optional[str] = None):
        """
        Initialize BatchNormalization layer.
        
        Args:
            axis: Axis to normalize
            momentum: Momentum for moving average
            epsilon: Small constant for numerical stability
            center: Whether to center the normalization
            scale: Whether to scale the normalization
            beta_initializer: Beta initialization method
            gamma_initializer: Gamma initialization method
            moving_mean_initializer: Moving mean initialization method
            moving_variance_initializer: Moving variance initialization method
            name: Optional name for the layer
        """
        super().__init__()
        
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.name = name or "batch_normalization"
        
        # Create PyTorch batch normalization layer
        self.bn = nn.BatchNorm1d(
            num_features=1,  # Will be set in build
            eps=epsilon,
            momentum=1 - momentum,
            affine=center and scale
        )
        
        self._built = False
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        # Determine the number of features
        if self.axis == -1:
            num_features = input_shape[-1]
        else:
            num_features = input_shape[self.axis]
        
        # Create PyTorch batch normalization layer
        self.bn = nn.BatchNorm1d(
            num_features=num_features,
            eps=self.epsilon,
            momentum=1 - self.momentum,
            affine=self.center and self.scale
        )
        
        self._built = True
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Normalized tensor
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Handle different input shapes
        if inputs.dim() == 2:
            # 2D input: (batch_size, features)
            return self.bn(inputs)
        elif inputs.dim() == 3:
            # 3D input: (batch_size, sequence_length, features)
            # Reshape to (batch_size * sequence_length, features)
            original_shape = inputs.shape
            inputs_reshaped = inputs.view(-1, inputs.size(-1))
            normalized = self.bn(inputs_reshaped)
            return normalized.view(original_shape)
        elif inputs.dim() == 4:
            # 4D input: (batch_size, channels, height, width)
            # Use BatchNorm2d for 4D inputs
            if not hasattr(self, 'bn_2d'):
                self.bn_2d = nn.BatchNorm2d(
                    num_features=inputs.size(1),
                    eps=self.epsilon,
                    momentum=1 - self.momentum,
                    affine=self.center and self.scale
                )
            return self.bn_2d(inputs)
        else:
            # For other dimensions, use BatchNorm1d
            return self.bn(inputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"BatchNormalization(axis={self.axis}, momentum={self.momentum}, epsilon={self.epsilon})"









