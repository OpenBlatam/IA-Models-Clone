"""
LSTM Layer for TruthGPT API
===========================

TensorFlow-like LSTM layer implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple


class LSTM(nn.Module):
    """
    LSTM layer for sequence processing.
    
    Similar to tf.keras.layers.LSTM, this layer applies
    Long Short-Term Memory processing to sequences.
    """
    
    def __init__(self, 
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 stateful: bool = False,
                 unroll: bool = False,
                 use_bias: bool = True,
                 dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 name: Optional[str] = None):
        """
        Initialize LSTM layer.
        
        Args:
            units: Number of LSTM units
            return_sequences: Whether to return sequences
            return_state: Whether to return states
            stateful: Whether to maintain state between batches
            unroll: Whether to unroll the LSTM
            use_bias: Whether to use bias
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            name: Optional name for the layer
        """
        super().__init__()
        
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.unroll = unroll
        self.use_bias = use_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.name = name or f"lstm_{units}"
        
        # Create PyTorch LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,  # Will be set in build
            hidden_size=units,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False
        )
        
        self._built = False
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        input_dim = input_shape[-1]
        
        # Update LSTM input size
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.units,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout if self.dropout > 0 else 0,
            bidirectional=False
        )
        
        self._built = True
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor(s)
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Forward pass through LSTM
        output, (hidden, cell) = self.lstm(inputs)
        
        if self.return_sequences and self.return_state:
            return output, hidden, cell
        elif self.return_sequences:
            return output
        elif self.return_state:
            return hidden
        else:
            # Return only the last output
            return output[:, -1, :]
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"LSTM(units={self.units}, return_sequences={self.return_sequences}, return_state={self.return_state})"









