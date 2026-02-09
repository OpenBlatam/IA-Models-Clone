"""
Sinusoidal Positional Encoding Module

Implements sinusoidal positional encoding (original Transformer).
"""

import logging
import math
import torch
import torch.nn as nn

from .base import PositionalEncoding

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Sinusoidal positional encoding (original Transformer).
    
    Args:
        embed_dim: Embedding dimension.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
    """
    
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__(embed_dim, dropout)
        self.max_len = max_len
        
        # Create positional encoding matrix
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        
        self.register_buffer('pe', pe)
        logger.debug(f"Initialized SinusoidalPositionalEncoding with embed_dim={embed_dim}, max_len={max_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Encoded tensor [batch_size, seq_len, embed_dim]
        """
        # Scale embeddings by sqrt(embed_dim)
        x = x * math.sqrt(self.embed_dim)
        
        seq_len = x.size(1)
        if seq_len > self.max_len:
            logger.warning(f"Sequence length {seq_len} exceeds max {self.max_len}")
            seq_len = self.max_len
            x = x[:, :seq_len, :]
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)



