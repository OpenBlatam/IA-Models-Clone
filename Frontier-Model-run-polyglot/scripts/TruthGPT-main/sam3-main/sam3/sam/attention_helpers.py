"""
Helper functions for attention mechanisms.
==========================================

Common utilities for attention operations:
- Flash attention handling
- Head separation/recombination
- Attention computation

Single Responsibility: Provide reusable attention utilities.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def separate_heads(x: Tensor, num_heads: int) -> Tensor:
    """
    Separate tensor into multiple attention heads.
    
    Args:
        x: Input tensor of shape (B, N, C)
        num_heads: Number of attention heads
        
    Returns:
        Tensor of shape (B, num_heads, N, C // num_heads)
    """
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head


def recombine_heads(x: Tensor) -> Tensor:
    """
    Recombine multiple attention heads into single tensor.
    
    Args:
        x: Tensor of shape (B, num_heads, N, C_per_head)
        
    Returns:
        Tensor of shape (B, N, num_heads * C_per_head)
    """
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C


def compute_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    use_fa3: bool = False
) -> Tensor:
    """
    Compute scaled dot-product attention with optional flash attention.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        dropout_p: Dropout probability
        use_fa3: Whether to use Flash Attention 3
        
    Returns:
        Attention output tensor
    """
    if use_fa3:
        from sam3.perflib.fa3 import flash_attn_func
        assert dropout_p == 0.0, "FA3 requires dropout_p=0.0"
        out = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    
    return out

