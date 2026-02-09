"""
Feed-Forward Network Implementation

Two-layer feed-forward network with activation.
"""

import torch
import torch.nn as nn
from typing import Callable


def get_activation(activation: str) -> nn.Module:
    """Get activation function."""
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),  # SiLU is Swish
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }
    
    if activation.lower() not in activation_map:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Available: {list(activation_map.keys())}"
        )
    
    return activation_map[activation.lower()]


class FeedForward(nn.Module):
    """
    Feed-forward network.
    
    Implements: FFN(x) = max(0, xW1 + b1)W2 + b2
    Or with GELU: FFN(x) = GELU(xW1 + b1)W2 + b2
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



