"""
Embedding Modules
Positional and token embeddings
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learnable positional encoding
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize token embedding
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            padding_idx: Padding index
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tokens: Token indices (batch, seq_len)
            
        Returns:
            Embeddings (batch, seq_len, embed_dim)
        """
        return self.embedding(tokens) * math.sqrt(self.embed_dim)


class EmbeddingLayer(nn.Module):
    """
    Complete embedding layer with token and positional encoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        use_learnable_pos: bool = False
    ):
        """
        Initialize embedding layer
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            padding_idx: Padding index
            use_learnable_pos: Use learnable positional encoding
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, padding_idx)
        
        if use_learnable_pos:
            self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len, dropout)
        else:
            self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tokens: Token indices
            
        Returns:
            Embedded tokens with positional encoding
        """
        x = self.token_embedding(tokens)
        x = self.pos_encoding(x)
        return x













