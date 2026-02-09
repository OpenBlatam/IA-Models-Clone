"""
Residual Feedforward Network Module

Implements feedforward network with residual connection and normalization.
"""

import logging
import torch
import torch.nn as nn

from .standard import FeedForward

logger = logging.getLogger(__name__)


class ResidualFeedForward(nn.Module):
    """
    Feedforward network with residual connection and normalization.
    
    Args:
        embed_dim: Embedding dimension.
        ff_dim: Feedforward hidden dimension.
        dropout: Dropout probability.
        activation: Activation function name.
        pre_norm: If True, apply normalization before feedforward (pre-norm).
                  If False, apply after (post-norm).
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.ff = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        logger.debug(f"Initialized ResidualFeedForward with embed_dim={embed_dim}, pre_norm={pre_norm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
        
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        residual = x
        
        if self.pre_norm:
            # Pre-norm: normalize before feedforward
            x = self.norm(x)
            x = self.ff(x)
            x = self.dropout(x)
            x = x + residual
        else:
            # Post-norm: normalize after feedforward
            x = self.ff(x)
            x = self.dropout(x)
            x = x + residual
            x = self.norm(x)
        
        return x



