"""
Functional Model for TruthGPT API
=================================

TensorFlow-like Functional model implementation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import Model


class Functional(Model, nn.Module):
    """
    Functional model for complex architectures.
    
    Similar to tf.keras.Model, this model allows you to create
    complex architectures with multiple inputs/outputs.
    """
    
    def __init__(self, 
                 inputs: Union[torch.Tensor, List[torch.Tensor]],
                 outputs: Union[torch.Tensor, List[torch.Tensor]],
                 name: Optional[str] = None):
        """
        Initialize Functional model.
        
        Args:
            inputs: Input tensor(s)
            outputs: Output tensor(s)
            name: Optional name for the model
        """
        super().__init__(name)
        nn.Module.__init__(self)
        
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        
        # Extract all layers from the computation graph
        self.layers = self._extract_layers()
    
    def _extract_layers(self) -> List[Any]:
        """Extract all layers from the computation graph."""
        layers = []
        visited = set()
        
        def traverse(tensor):
            if tensor in visited:
                return
            
            visited.add(tensor)
            
            if hasattr(tensor, 'grad_fn') and tensor.grad_fn is not None:
                # This is a computed tensor, find its operation
                if hasattr(tensor.grad_fn, 'next_functions'):
                    for next_fn in tensor.grad_fn.next_functions:
                        if next_fn[0] is not None:
                            traverse(next_fn[0])
            
            # Check if this tensor has a layer attribute
            if hasattr(tensor, 'layer'):
                layers.append(tensor.layer)
        
        for output in self.outputs:
            traverse(output)
        
        return layers
    
    def call(self, inputs: Union[torch.Tensor, List[torch.Tensor]], 
             training: bool = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensor(s)
            training: Whether in training mode
            
        Returns:
            Output tensor(s)
        """
        if isinstance(inputs, list):
            # Multiple inputs
            if len(inputs) != len(self.inputs):
                raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(inputs)}")
            
            # Process each input through the computation graph
            results = []
            for i, inp in enumerate(inputs):
                # This is a simplified version - in practice, you'd need
                # to trace the computation graph properly
                result = self._forward_single(inp, training)
                results.append(result)
            
            return results if len(results) > 1 else results[0]
        else:
            # Single input
            return self._forward_single(inputs, training)
    
    def _forward_single(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """Forward pass for single input."""
        # This is a simplified implementation
        # In practice, you'd need to properly trace the computation graph
        x = inputs
        
        for layer in self.layers:
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
        print(f"Number of inputs: {len(self.inputs)}")
        print(f"Number of outputs: {len(self.outputs)}")
        print(f"Number of layers: {len(self.layers)}")
        print("\nLayer details:")
        for i, layer in enumerate(self.layers):
            print(f"  {i}: {layer}")
    
    def __len__(self):
        """Return number of layers."""
        return len(self.layers)
    
    def __getitem__(self, index):
        """Get layer by index."""
        return self.layers[index]









