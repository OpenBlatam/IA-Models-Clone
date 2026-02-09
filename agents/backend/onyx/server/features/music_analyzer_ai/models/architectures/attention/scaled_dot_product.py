"""
Scaled Dot-Product Attention

Implements the standard scaled dot-product attention mechanism
following "Attention is All You Need" paper.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with proper scaling and masking.
    
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Following PyTorch best practices:
    - Efficient matrix operations
    - Proper masking support
    - Dropout for regularization
    """
    
    def __init__(
        self,
        head_dim: int,
        dropout: float = 0.1,
        scale: Optional[float] = None
    ):
        """
        Initialize scaled dot-product attention.
        
        Args:
            head_dim: Dimension of each attention head
            dropout: Dropout probability
            scale: Optional custom scale (defaults to 1/sqrt(head_dim))
        """
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention.
        
        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, num_heads, seq_len, head_dim]
            value: Value tensor [batch, num_heads, seq_len, head_dim]
            mask: Optional attention mask [batch, seq_len, seq_len]
            attn_mask: Optional attention mask [batch, num_heads, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, num_heads, seq_len, head_dim]
            - attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if mask is not None:
            # Expand mask to match attention scores shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


__all__ = [
    "ScaledDotProductAttention",
]



