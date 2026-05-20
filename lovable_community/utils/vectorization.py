"""
Vectorization Optimizations

Highly optimized vectorized operations for maximum speed.
"""

import torch
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


def vectorized_embedding_lookup(
    embeddings: torch.Tensor,
    indices: torch.Tensor
) -> torch.Tensor:
    """
    Optimized embedding lookup using advanced indexing.
    
    Args:
        embeddings: Embedding matrix [vocab_size, dim]
        indices: Token indices [batch_size, seq_len]
        
    Returns:
        Embedded vectors [batch_size, seq_len, dim]
    """
    # Use advanced indexing for faster lookup
    return embeddings[indices]


def vectorized_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Vectorized batch normalization.
    
    Args:
        x: Input tensor
        running_mean: Running mean
        running_var: Running variance
        weight: Scale parameter
        bias: Shift parameter
        eps: Epsilon
        
    Returns:
        Normalized tensor
    """
    # Vectorized normalization
    normalized = (x - running_mean) / torch.sqrt(running_var + eps)
    
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    
    return normalized


def vectorized_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Optimized softmax with numerical stability.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability (vectorized)
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def vectorized_matmul_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Highly optimized matrix multiplication for attention.
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_k]
        k: Key tensor [batch, heads, seq_len, d_k]
        v: Value tensor [batch, heads, seq_len, d_k]
        scale: Scaling factor
        mask: Attention mask
        
    Returns:
        Attention output
    """
    # Optimized matmul
    scores = torch.bmm(q.view(-1, q.size(2), q.size(3)),
                      k.view(-1, k.size(2), k.size(3)).transpose(1, 2)) * scale
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = vectorized_softmax(scores, dim=-1)
    output = torch.bmm(attn_weights, v.view(-1, v.size(2), v.size(3)))
    
    return output.view(q.shape[0], q.shape[1], q.shape[2], -1)


def vectorized_gather(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """
    Optimized gather operation.
    
    Args:
        input_tensor: Source tensor
        indices: Indices to gather
        dim: Dimension to gather along
        
    Returns:
        Gathered tensor
    """
    return torch.gather(input_tensor, dim, indices)


def vectorized_scatter(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0,
    value: float = 0.0
) -> torch.Tensor:
    """
    Optimized scatter operation.
    
    Args:
        input_tensor: Target tensor
        indices: Indices to scatter
        dim: Dimension to scatter along
        value: Value to scatter
        
    Returns:
        Scattered tensor
    """
    return torch.scatter(input_tensor, dim, indices, value)


class VectorizedOperations:
    """
    Collection of vectorized operations for maximum speed.
    """
    
    @staticmethod
    def batch_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Vectorized batch dot product."""
        return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()
    
    @staticmethod
    def batch_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Vectorized batch cosine similarity."""
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return (x_norm * y_norm).sum(dim=-1)
    
    @staticmethod
    def batch_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Vectorized batch euclidean distance."""
        return torch.cdist(x, y, p=2)
    
    @staticmethod
    def batch_l2_norm(x: torch.Tensor) -> torch.Tensor:
        """Vectorized batch L2 norm."""
        return torch.norm(x, p=2, dim=-1)













