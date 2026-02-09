"""
Advanced compression techniques for KV Cache.

Provides additional compression methods beyond basic compression.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from kv_cache.types import TensorPair
from kv_cache.interfaces import ICompressor

logger = logging.getLogger(__name__)


class SVDCompressor(ICompressor):
    """
    SVD-based compression for KV cache.
    
    Uses Singular Value Decomposition for efficient compression.
    """
    
    def __init__(self, rank: int, use_amp: bool = True):
        """
        Initialize SVD compressor.
        
        Args:
            rank: Rank for SVD (lower = more compression)
            use_amp: Whether to use mixed precision
        """
        self.rank = rank
        self.use_amp = use_amp
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> TensorPair:
        """
        Compress using SVD.
        
        Args:
            key: Key tensor
            value: Value tensor
            dtype: Target dtype
            
        Returns:
            Compressed (key, value) pair
        """
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp and key.is_cuda):
                # SVD compression
                key_compressed = self._svd_compress(key, self.rank)
                value_compressed = self._svd_compress(value, self.rank)
                
                return key_compressed, value_compressed
        except Exception as e:
            logger.warning(f"SVD compression failed: {e}")
            return key, value
    
    def _svd_compress(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Apply SVD compression to tensor."""
        if tensor.dim() < 2:
            return tensor
        
        # Reshape to 2D if needed
        original_shape = tensor.shape
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor
        
        # Compute SVD
        U, S, V = torch.linalg.svd(tensor_2d, full_matrices=False)
        
        # Truncate to rank
        rank = min(rank, S.shape[0])
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        V_trunc = V[:rank, :]
        
        # Reconstruct
        compressed = U_trunc @ torch.diag(S_trunc) @ V_trunc
        
        # Reshape back
        if len(original_shape) > 2:
            compressed = compressed.view(original_shape)
        
        return compressed


class LowRankCompressor(ICompressor):
    """
    Low-rank approximation compression.
    
    Uses matrix factorization for compression.
    """
    
    def __init__(self, rank: int, use_amp: bool = True):
        """
        Initialize low-rank compressor.
        
        Args:
            rank: Rank for approximation
            use_amp: Whether to use mixed precision
        """
        self.rank = rank
        self.use_amp = use_amp
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> TensorPair:
        """
        Compress using low-rank approximation.
        
        Args:
            key: Key tensor
            value: Value tensor
            dtype: Target dtype
            
        Returns:
            Compressed (key, value) pair
        """
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp and key.is_cuda):
                key_compressed = self._lowrank_compress(key, self.rank)
                value_compressed = self._lowrank_compress(value, self.rank)
                
                return key_compressed, value_compressed
        except Exception as e:
            logger.warning(f"Low-rank compression failed: {e}")
            return key, value
    
    def _lowrank_compress(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Apply low-rank compression."""
        if tensor.dim() < 2:
            return tensor
        
        original_shape = tensor.shape
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor
        
        # Factorize: A ≈ U @ V^T
        U, S, V = torch.linalg.svd(tensor_2d, full_matrices=False)
        
        rank = min(rank, S.shape[0])
        U_factor = U[:, :rank] @ torch.diag(S[:rank])
        V_factor = V[:rank, :]
        
        # Reconstruct
        compressed = U_factor @ V_factor
        
        if len(original_shape) > 2:
            compressed = compressed.view(original_shape)
        
        return compressed


class BlockSparseCompressor(ICompressor):
    """
    Block-sparse compression.
    
    Removes blocks of zeros or low-magnitude values.
    """
    
    def __init__(self, sparsity: float = 0.5, block_size: int = 8, use_amp: bool = True):
        """
        Initialize block-sparse compressor.
        
        Args:
            sparsity: Target sparsity (0.0 = no compression, 1.0 = maximum)
            block_size: Size of blocks to sparsify
            use_amp: Whether to use mixed precision
        """
        self.sparsity = sparsity
        self.block_size = block_size
        self.use_amp = use_amp
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> TensorPair:
        """
        Compress using block sparsity.
        
        Args:
            key: Key tensor
            value: Value tensor
            dtype: Target dtype
            
        Returns:
            Compressed (key, value) pair
        """
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp and key.is_cuda):
                key_compressed = self._block_sparse(key, self.sparsity, self.block_size)
                value_compressed = self._block_sparse(value, self.sparsity, self.block_size)
                
                return key_compressed, value_compressed
        except Exception as e:
            logger.warning(f"Block-sparse compression failed: {e}")
            return key, value
    
    def _block_sparse(
        self,
        tensor: torch.Tensor,
        sparsity: float,
        block_size: int
    ) -> torch.Tensor:
        """Apply block-sparse compression."""
        if sparsity <= 0.0:
            return tensor
        
        # Calculate threshold
        sorted_values = torch.abs(tensor.flatten()).sort(descending=True)[0]
        threshold_idx = int(len(sorted_values) * sparsity)
        threshold = sorted_values[threshold_idx] if threshold_idx < len(sorted_values) else 0.0
        
        # Create mask for blocks
        mask = torch.abs(tensor) > threshold
        compressed = tensor * mask
        
        return compressed

