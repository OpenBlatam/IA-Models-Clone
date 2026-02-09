"""
Attention Modules
Reusable attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Use bias
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention module
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize self-attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of heads
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        
        # Residual connection and layer norm
        out = self.norm(x + self.dropout(attn_out))
        
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention module
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of heads
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        attn_out = self.attention(query, key, value, mask)
        out = self.norm(query + self.dropout(attn_out))
        return out













