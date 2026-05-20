"""
Cache operations module.

Centralizes common cache operations for better organization.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from kv_cache.types import TensorPair, StatsDict
from kv_cache.exceptions import CacheValidationError, CacheOperationError

logger = logging.getLogger(__name__)


class CacheOperations:
    """
    Centralized cache operations.
    
    Provides high-level operations that combine multiple components.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache operations.
        
        Args:
            cache: Cache instance (BaseKVCache or compatible)
        """
        self.cache = cache
    
    def get_or_compute(
        self,
        position: int,
        compute_fn: Any,
        *args: Any,
        **kwargs: Any
    ) -> TensorPair:
        """
        Get from cache or compute if missing.
        
        Args:
            position: Cache position
            compute_fn: Function to compute value if cache miss
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
            
        Returns:
            Tuple of (key, value) from cache or computation
        """
        # Try to get from cache
        cached = self.cache.get(position)
        if cached is not None:
            return cached
        
        # Cache miss - compute
        result = compute_fn(*args, **kwargs)
        
        # Ensure result is TensorPair
        if not isinstance(result, tuple) or len(result) != 2:
            raise CacheOperationError(
                "compute_fn must return (key, value) tuple"
            )
        
        key, value = result
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise CacheOperationError(
                "compute_fn must return torch.Tensor tuple"
            )
        
        # Store in cache
        self.cache.put(position, key, value)
        
        return key, value
    
    def batch_get_or_compute(
        self,
        positions: list[int],
        compute_fn: Any,
        *args: Any,
        **kwargs: Any
    ) -> list[TensorPair]:
        """
        Batch get or compute.
        
        Args:
            positions: List of cache positions
            compute_fn: Function to compute values (should handle batch)
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
            
        Returns:
            List of (key, value) tuples
        """
        results = []
        
        # Try to get all from cache
        cache_misses = []
        for pos in positions:
            cached = self.cache.get(pos)
            if cached is not None:
                results.append(cached)
            else:
                cache_misses.append(pos)
        
        # Compute missing values
        if cache_misses:
            computed = compute_fn(*args, **kwargs)
            # Assume compute_fn returns batch
            if isinstance(computed, tuple) and len(computed) == 2:
                keys, values = computed
                for i, pos in enumerate(cache_misses):
                    key = keys[i] if keys.dim() > 0 else keys
                    value = values[i] if values.dim() > 0 else values
                    self.cache.put(pos, key, value)
                    results.append((key, value))
        
        return results
    
    def update_entry(
        self,
        position: int,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None
    ) -> bool:
        """
        Update existing cache entry.
        
        Args:
            position: Cache position
            key: Optional new key (None = keep existing)
            value: Optional new value (None = keep existing)
            
        Returns:
            True if entry was updated, False if not found
        """
        cached = self.cache.get(position)
        if cached is None:
            return False
        
        old_key, old_value = cached
        new_key = key if key is not None else old_key
        new_value = value if value is not None else old_value
        
        self.cache.put(position, new_key, new_value)
        return True
    
    def get_or_default(
        self,
        position: int,
        default_key: torch.Tensor,
        default_value: torch.Tensor
    ) -> TensorPair:
        """
        Get from cache or return default if missing.
        
        Args:
            position: Cache position
            default_key: Default key if cache miss
            default_value: Default value if cache miss
            
        Returns:
            Tuple of (key, value)
        """
        cached = self.cache.get(position)
        if cached is not None:
            return cached
        return default_key, default_value
    
    def evict_oldest(self, num_to_evict: int = 1) -> int:
        """
        Evict oldest entries (helper for common operation).
        
        Args:
            num_to_evict: Number of entries to evict
            
        Returns:
            Number of entries actually evicted
        """
        # Get access times
        access_times = self.cache.storage.get_access_times()
        if not access_times:
            return 0
        
        # Sort by access time (oldest first)
        sorted_positions = sorted(
            access_times.items(),
            key=lambda x: x[1]
        )[:num_to_evict]
        
        positions_to_evict = [pos for pos, _ in sorted_positions]
        return self.cache.storage.remove(positions_to_evict)
    
    def warm_cache(
        self,
        positions: list[int],
        compute_fn: Any,
        *args: Any,
        **kwargs: Any
    ) -> int:
        """
        Warm cache with precomputed values.
        
        Args:
            positions: List of positions to warm
            compute_fn: Function to compute values
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
            
        Returns:
            Number of entries warmed
        """
        warmed = 0
        for pos in positions:
            if self.cache.get(pos) is None:
                result = compute_fn(pos, *args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    key, value = result
                    self.cache.put(pos, key, value)
                    warmed += 1
        return warmed



