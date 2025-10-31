"""
Dropout Layer for TruthGPT API
=============================

TensorFlow-like Dropout layer implementation.
"""

import torch
import torch.nn as nn
from typing import Optional


class Dropout(nn.Module):
    """
    Dropout layer for regularization.
    
    Similar to tf.keras.layers.Dropout, this layer randomly sets
    input units to 0 during training to prevent overfitting.
    """
    
    def __init__(self, 
                 rate: float = 0.5,
                 noise_shape: Optional[tuple] = None,
                 seed: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialize Dropout layer.
        
        Args:
            rate: Fraction of input units to drop
            noise_shape: Shape of the noise tensor
            seed: Random seed
            name: Optional name for the layer
        """
        super().__init__()
        
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.name = name or f"dropout_{rate}"
        
        # Create PyTorch dropout layer
        self.dropout = nn.Dropout(p=rate)
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        if training is None:
            training = self.training
        
        if training:
            return self.dropout(inputs)
        else:
            return inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"Dropout(rate={self.rate})"









