"""
Memory pool for KV Cache.

Provides efficient memory allocation and reuse for cache tensors.
"""
from __future__ import annotations

import logging
import torch
from typing import Optional, Dict, List, Tuple
from collections import deque

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class TensorMemoryPool:
    """
    Memory pool for reusing tensor allocations.
    
    Reduces memory fragmentation and allocation overhead.
    """
    
    def __init__(
        self,
        max_pool_size: int = 100,
        enabled: bool = True
    ):
        """
        Initialize memory pool.
        
        Args:
            max_pool_size: Maximum number of tensors to keep in pool
            enabled: Whether pooling is enabled
        """
        self.max_pool_size = max_pool_size
        self.enabled = enabled
        
        # Pool: shape -> deque of tensors
        self._pools: Dict[Tuple[int, ...], deque] = {}
        self._total_allocated = 0
        self._total_reused = 0
    
    def get_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get tensor from pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            device: Tensor device
            
        Returns:
            Tensor from pool or newly allocated
        """
        if not self.enabled:
            return torch.empty(shape, dtype=dtype, device=device)
        
        key = (*shape, dtype, device)
        
        if key in self._pools and self._pools[key]:
            # Reuse from pool
            tensor = self._pools[key].popleft()
            self._total_reused += 1
            # Zero out tensor for clean state
            tensor.zero_()
            return tensor
        else:
            # Allocate new
            self._total_allocated += 1
            return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if not self.enabled:
            return
        
        key = (*tensor.shape, tensor.dtype, tensor.device)
        
        if key not in self._pools:
            self._pools[key] = deque(maxlen=self.max_pool_size)
        
        # Only pool if pool not full
        if len(self._pools[key]) < self.max_pool_size:
            # Detach and clear gradients
            tensor = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
            self._pools[key].append(tensor)
    
    def clear_pool(self) -> None:
        """Clear all pooled tensors."""
        self._pools.clear()
        logger.debug("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool stats
        """
        total_pooled = sum(len(pool) for pool in self._pools.values())
        return {
            "total_allocated": self._total_allocated,
            "total_reused": self._total_reused,
            "total_pooled": total_pooled,
            "pool_size": len(self._pools),
            "reuse_rate": (
                self._total_reused / max(self._total_allocated, 1)
            ) * 100
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_pool()


class KVCacheMemoryPool:
    """
    Memory pool specifically for KV cache entries.
    
    Optimized for key-value tensor pairs.
    """
    
    def __init__(
        self,
        max_pool_size: int = 50,
        enabled: bool = True
    ):
        """
        Initialize KV cache memory pool.
        
        Args:
            max_pool_size: Maximum number of KV pairs to keep
            enabled: Whether pooling is enabled
        """
        self.max_pool_size = max_pool_size
        self.enabled = enabled
        
        # Pool for (key_shape, value_shape) -> deque of (key, value)
        self._pools: Dict[Tuple, deque] = {}
        self._total_allocated = 0
        self._total_reused = 0
    
    def get_kv_pair(
        self,
        key_shape: Tuple[int, ...],
        value_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None
    ) -> TensorPair:
        """
        Get KV pair from pool or allocate new.
        
        Args:
            key_shape: Key tensor shape
            value_shape: Value tensor shape
            dtype: Tensor dtype
            device: Tensor device
            
        Returns:
            Tuple of (key, value) tensors
        """
        if not self.enabled:
            key = torch.empty(key_shape, dtype=dtype, device=device)
            value = torch.empty(value_shape, dtype=dtype, device=device)
            return key, value
        
        key = (key_shape, value_shape, dtype, device)
        
        if key in self._pools and self._pools[key]:
            # Reuse from pool
            key_tensor, value_tensor = self._pools[key].popleft()
            self._total_reused += 1
            key_tensor.zero_()
            value_tensor.zero_()
            return key_tensor, value_tensor
        else:
            # Allocate new
            self._total_allocated += 1
            key_tensor = torch.empty(key_shape, dtype=dtype, device=device)
            value_tensor = torch.empty(value_shape, dtype=dtype, device=device)
            return key_tensor, value_tensor
    
    def return_kv_pair(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Return KV pair to pool.
        
        Args:
            key: Key tensor
            value: Value tensor
        """
        if not self.enabled:
            return
        
        pool_key = (key.shape, value.shape, key.dtype, key.device)
        
        if pool_key not in self._pools:
            self._pools[pool_key] = deque(maxlen=self.max_pool_size)
        
        if len(self._pools[pool_key]) < self.max_pool_size:
            # Detach and move to CPU if needed
            key = key.detach().cpu() if key.is_cuda else key.detach()
            value = value.detach().cpu() if value.is_cuda else value.detach()
            self._pools[pool_key].append((key, value))
    
    def clear_pool(self) -> None:
        """Clear all pooled KV pairs."""
        self._pools.clear()
        logger.debug("KV cache memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool stats
        """
        total_pooled = sum(len(pool) for pool in self._pools.values())
        return {
            "total_allocated": self._total_allocated,
            "total_reused": self._total_reused,
            "total_pooled": total_pooled,
            "pool_size": len(self._pools),
            "reuse_rate": (
                self._total_reused / max(self._total_allocated, 1)
            ) * 100
        }

