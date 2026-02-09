"""
Conv2D Layer for TruthGPT API
============================

TensorFlow-like Conv2D layer implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple


class Conv2D(nn.Module):
    """
    2D Convolutional layer.
    
    Similar to tf.keras.layers.Conv2D, this layer applies 2D convolution
    to the input followed by an optional activation function.
    """
    
    def __init__(self, 
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[str, int, Tuple[int, int]] = 'valid',
                 activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        """
        Initialize Conv2D layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of the convolution kernel
            strides: Stride of the convolution
            padding: Padding mode
            activation: Activation function
            use_bias: Whether to use bias
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            name: Optional name for the layer
        """
        super().__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.use_bias = use_bias
        self.name = name or f"conv2d_{filters}"
        
        # Initialize activation function
        self.activation = self._get_activation(activation)
        
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
        
        input_channels = input_shape[1]  # Assuming NCHW format
        
        # Create PyTorch Conv2D layer
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._get_padding(),
            bias=self.use_bias
        )
        
        # Initialize weights
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        elif self.kernel_initializer == 'zeros':
            nn.init.zeros_(self.conv.weight)
        elif self.kernel_initializer == 'ones':
            nn.init.ones_(self.conv.weight)
        
        # Initialize bias
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                nn.init.zeros_(self.conv.bias)
            elif self.bias_initializer == 'ones':
                nn.init.ones_(self.conv.bias)
        
        self._built = True
    
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
        if not self._built:
            self.build(inputs.shape)
        
        # Apply convolution
        output = self.conv(inputs)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"Conv2D(filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding})"









