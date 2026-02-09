"""
Multi-Head Attention

Implements multi-head attention mechanism with proper
head splitting and concatenation.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Splits input into multiple heads, applies attention to each,
    then concatenates and projects the results.
    
    Following PyTorch best practices:
    - Efficient linear projections
    - Proper head splitting
    - Residual connections support
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            kdim: Key dimension (defaults to embed_dim)
            vdim: Value dimension (defaults to embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(
            head_dim=self.head_dim,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, kdim]
            value: Value tensor [batch, seq_len, vdim]
            mask: Optional attention mask
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
            Optionally returns attention weights if return_attention=True
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head: [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            Q, K, V,
            mask=mask,
            attn_mask=attn_mask
        )
        
        # Concatenate heads: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output


__all__ = [
    "MultiHeadAttention",
]

