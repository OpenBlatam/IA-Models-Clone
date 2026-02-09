"""
Modular Positional Encoding
Various positional encoding strategies for sequence models
"""

import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class PositionalEncoding(nn.Module):
    """
    Base class for positional encodings
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding"""
        raise NotImplementedError


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Sinusoidal positional encoding (original Transformer)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            [batch_size, seq_len, embed_dim]
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


class LearnedPositionalEncoding(PositionalEncoding):
    """
    Learned positional encoding (parameterized)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            [batch_size, seq_len, embed_dim]
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


# Alias for backward compatibility
PositionalEncoding = SinusoidalPositionalEncoding



