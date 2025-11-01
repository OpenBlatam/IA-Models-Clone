"""
Compression module for KV Cache.

Provides various compression strategies for memory efficiency.
"""
import logging
from typing import Tuple, Optional
import torch
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class Compressor:
    """
    Handles compression of key-value tensors.
    
    Supports multiple compression methods:
    - SVD (Singular Value Decomposition)
    - Low-rank approximation
    - Sparse compression
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.3,
        method: str = "svd",
        use_amp: bool = False
    ):
        """
        Initialize compressor.
        
        Args:
            compression_ratio: Target compression ratio (0 < ratio <= 1)
            method: Compression method (svd|lowrank|sparse)
            use_amp: Whether to use automatic mixed precision
        """
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        
        if method not in ["svd", "lowrank", "sparse"]:
            raise ValueError(f"method must be svd, lowrank, or sparse, got {method}")
        
        self.compression_ratio = compression_ratio
        self.method = method
        self.use_amp = use_amp
        logger.info(f"Initialized Compressor with ratio={compression_ratio}, method={method}")
    
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress key-value tensors.
        
        Args:
            key: Key tensor to compress
            value: Value tensor to compress
            dtype: Optional dtype for autocast
            
        Returns:
            Compressed (key, value) tensors
        """
        if self.compression_ratio >= 1.0:
            return key, value
        
        try:
            with autocast(enabled=self.use_amp, dtype=dtype):
                if self.method == "svd":
                    return self._compress_svd(key, value)
                elif self.method == "lowrank":
                    return self._compress_lowrank(key, value)
                elif self.method == "sparse":
                    return self._compress_sparse(key, value)
                else:
                    logger.warning(f"Unknown compression method {self.method}, using SVD")
                    return self._compress_svd(key, value)
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using uncompressed")
            return key, value
    
    def _compress_svd(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress using SVD (Singular Value Decomposition).
        
        This is a simplified implementation. For production,
        use proper SVD with stored components for decompression.
        """
        # Flatten tensors
        key_flat = key.flatten()
        value_flat = value.flatten()
        
        # Calculate target size
        compressed_size = int(len(key_flat) * self.compression_ratio)
        
        # Simplified: just truncate (real implementation would use SVD)
        if compressed_size < len(key_flat):
            key_compressed = key_flat[:compressed_size]
            value_compressed = value_flat[:compressed_size]
            
            # Try to reshape if possible
            if key_compressed.numel() == key.numel():
                return key, value
            
            return key_compressed.reshape(-1), value_compressed.reshape(-1)
        
        return key, value
    
    def _compress_lowrank(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using low-rank approximation."""
        # Simplified low-rank compression
        # In production, use proper low-rank decomposition
        return self._compress_svd(key, value)
    
    def _compress_sparse(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using sparsity (keep top-k values)."""
        # Simplified: keep top values
        k = int(key.numel() * self.compression_ratio)
        if k < key.numel():
            key_topk, key_indices = torch.topk(key.abs().flatten(), k)
            value_topk = value.flatten()[key_indices]
            return key_topk, value_topk
        return key, value

