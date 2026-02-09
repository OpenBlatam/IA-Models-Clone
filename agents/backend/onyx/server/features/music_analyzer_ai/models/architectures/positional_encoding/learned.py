"""
Learned Positional Encoding Module

Implements learned positional encoding (parameterized).
"""

import logging
import math
import torch
import torch.nn as nn

from .base import PositionalEncoding

logger = logging.getLogger(__name__)


class LearnedPositionalEncoding(PositionalEncoding):
    """
    Learned positional encoding (parameterized).
    
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
        
        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.empty(1, max_len, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        logger.debug(f"Initialized LearnedPositionalEncoding with embed_dim={embed_dim}, max_len={max_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Encoded tensor [batch_size, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            logger.warning(f"Sequence length {seq_len} exceeds max {self.max_len}")
            seq_len = self.max_len
            x = x[:, :seq_len, :]
        
        # Scale embeddings by sqrt(embed_dim)
        x = x * math.sqrt(self.embed_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        return self.dropout(x)



