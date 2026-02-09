"""
Base Positional Encoding Module

Base class for positional encodings.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Base class for positional encodings.
    
    Args:
        embed_dim: Embedding dimension.
        dropout: Dropout probability.
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Encoded tensor [batch_size, seq_len, embed_dim]
        """
        raise NotImplementedError("Subclasses must implement forward method")



