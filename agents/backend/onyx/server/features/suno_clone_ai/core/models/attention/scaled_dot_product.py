"""
Scaled Dot-Product Attention

Core attention mechanism implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, scale: float, dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            scale: Scaling factor (typically sqrt(d_k))
            dropout: Dropout probability
        """
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch_size, num_heads, seq_len_q, d_k)
            key: Key tensor (batch_size, num_heads, seq_len_k, d_k)
            value: Value tensor (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional attention mask
            
        Returns:
            Tuple of (context, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights



