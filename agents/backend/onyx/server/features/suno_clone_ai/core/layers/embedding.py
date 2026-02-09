"""
Embedding Layers

Implements positional encodings and token embeddings.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Base positional encoding."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        seq_len = x.size(0)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
        pos_emb = self.embedding(positions).transpose(0, 1)
        x = x + pos_emb
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding with optional positional encoding."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        positional_encoding: Optional[str] = None,
        max_len: int = 5000
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )
        
        if positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        elif positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len)
        else:
            self.pos_encoding = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape (batch, seq_len)
            
        Returns:
            Embedded tokens of shape (seq_len, batch, d_model)
        """
        x = self.embedding(x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        
        if self.pos_encoding:
            x = self.pos_encoding(x)
        
        return x



