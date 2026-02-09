"""
Sequential Model for TruthGPT API
=================================

TensorFlow-like Sequential model implementation.
"""

import torch
import torch.nn as nn
from typing import List, Union, Any, Optional
from .base import Model


class Sequential(Model, nn.Module):
    """
    Sequential model for stacking layers.
    
    Similar to tf.keras.Sequential, this model allows you to stack
    layers in a linear fashion.
    """
    
    def __init__(self, layers: List[Any] = None, name: Optional[str] = None):
        """
        Initialize Sequential model.
        
        Args:
            layers: List of layers to add
            name: Optional name for the model
        """
        super().__init__(name)
        nn.Module.__init__(self)
        
        self.layers_list = nn.ModuleList()
        
        if layers:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer: Any):
        """
        Add a layer to the model.
        
        Args:
            layer: Layer to add
        """
        if hasattr(layer, 'build'):
            # If layer has build method, call it
            layer.build()
        
        self.layers_list.append(layer)
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = inputs
        
        for layer in self.layers_list:
            if hasattr(layer, 'call'):
                x = layer.call(x, training=training)
            elif hasattr(layer, '__call__'):
                x = layer(x)
            else:
                x = layer(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def summary(self):
        """Print model summary."""
        super().summary()
        print(f"Number of layers: {len(self.layers_list)}")
        print("\nLayer details:")
        for i, layer in enumerate(self.layers_list):
            print(f"  {i}: {layer}")
    
    def __len__(self):
        """Return number of layers."""
        return len(self.layers_list)
    
    def __getitem__(self, index):
        """Get layer by index."""
        return self.layers_list[index]









