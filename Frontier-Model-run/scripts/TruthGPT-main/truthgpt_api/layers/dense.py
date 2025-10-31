"""
Dense Layer for TruthGPT API
===========================

TensorFlow-like Dense layer implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable


class Dense(nn.Module):
    """
    Dense (fully connected) layer.
    
    Similar to tf.keras.layers.Dense, this layer applies a linear transformation
    to the input followed by an optional activation function.
    """
    
    def __init__(self, 
                 units: int,
                 activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        """
        Initialize Dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function
            use_bias: Whether to use bias
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            name: Optional name for the layer
        """
        super().__init__()
        
        self.units = units
        self.use_bias = use_bias
        self.name = name or f"dense_{units}"
        
        # Initialize activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(units, 1))  # Will be resized in build
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(units))
        else:
            self.bias = None
        
        # Store initialization methods
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        self._built = False
    
    def _get_activation(self, activation: Optional[Union[str, Callable]]):
        """Get activation function."""
        if activation is None:
            return None
        elif isinstance(activation, str):
            activations = {
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'tanh': nn.Tanh(),
                'softmax': nn.Softmax(dim=-1),
                'linear': None
            }
            return activations.get(activation, None)
        elif callable(activation):
            return activation()
        else:
            return None
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        input_dim = input_shape[-1]
        
        # Initialize weight
        if self.kernel_initializer == 'glorot_uniform':
            self.weight = nn.Parameter(torch.randn(self.units, input_dim) * (2.0 / (input_dim + self.units))**0.5)
        elif self.kernel_initializer == 'zeros':
            self.weight = nn.Parameter(torch.zeros(self.units, input_dim))
        elif self.kernel_initializer == 'ones':
            self.weight = nn.Parameter(torch.ones(self.units, input_dim))
        else:
            self.weight = nn.Parameter(torch.randn(self.units, input_dim))
        
        # Initialize bias
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.bias = nn.Parameter(torch.zeros(self.units))
            elif self.bias_initializer == 'ones':
                self.bias = nn.Parameter(torch.ones(self.units))
            else:
                self.bias = nn.Parameter(torch.zeros(self.units))
        
        self._built = True
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Linear transformation
        output = torch.matmul(inputs, self.weight.t())
        
        # Add bias
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"Dense(units={self.units}, activation={self.activation}, use_bias={self.use_bias})"









