"""
Transformer Block Implementation

Complete transformer block with attention and feed-forward layers.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    
    Implements proper normalization (LayerNorm) and residual connections.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swish')
            layer_norm_eps: LayerNorm epsilon
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

