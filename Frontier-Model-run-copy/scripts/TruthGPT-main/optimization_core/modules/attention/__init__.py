"""
Attention module for TruthGPT Optimization Core
Contains multi-head attention and specialized attention implementations
"""

from .multi_head_attention import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    create_multi_head_attention
)

from .flash_attention import (
    FlashAttention,
    FlashAttentionV2,
    create_flash_attention
)

from .sparse_attention import (
    SparseAttention,
    LocalAttention,
    StridedAttention,
    create_sparse_attention
)

from .cross_attention import (
    CrossAttention,
    create_cross_attention
)

__all__ = [
    # Multi-Head Attention
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    'create_multi_head_attention',
    
    # Flash Attention
    'FlashAttention',
    'FlashAttentionV2',
    'create_flash_attention',
    
    # Sparse Attention
    'SparseAttention',
    'LocalAttention',
    'StridedAttention',
    'create_sparse_attention',
    
    # Cross Attention
    'CrossAttention',
    'create_cross_attention'
]


