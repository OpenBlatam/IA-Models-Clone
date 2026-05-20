"""
Batch processor for KV Cache.

Optimized batch operations for high throughput.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Callable
import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for efficient cache operations.
    
    Optimizes batch operations for better throughput.
    """
    
    def __init__(
        self,
        cache: Any,
        batch_size: int = 32,
        use_async: bool = False
    ):
        """
        Initialize batch processor.
        
        Args:
            cache: Cache instance
            batch_size: Default batch size
            use_async: Whether to use async operations
        """
        self.cache = cache
        self.batch_size = batch_size
        self.use_async = use_async
    
    def batch_get(
        self,
        positions: List[int],
        batch_size: Optional[int] = None
    ) -> List[Optional[TensorPair]]:
        """
        Batch get operation.
        
        Args:
            positions: List of positions to get
            batch_size: Batch size (None = use default)
            
        Returns:
            List of cached entries
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            batch_results = [self.cache.get(pos) for pos in batch]
            results.extend(batch_results)
        
        return results
    
    def batch_put(
        self,
        entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Batch put operation.
        
        Args:
            entries: List of (position, key, value) tuples
            batch_size: Batch size (None = use default)
            
        Returns:
            Number of entries stored
        """
        batch_size = batch_size or self.batch_size
        stored = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            for pos, key, value in batch:
                self.cache.put(pos, key, value)
                stored += 1
        
        return stored
    
    def batch_get_or_compute(
        self,
        positions: List[int],
        compute_fn: Callable[[List[int]], Tuple[torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> List[TensorPair]:
        """
        Batch get or compute.
        
        Args:
            positions: List of positions
            compute_fn: Function that computes batch (keys, values)
            batch_size: Batch size (None = use default)
            
        Returns:
            List of (key, value) pairs
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        # First, try to get from cache
        cache_hits = {}
        cache_misses = []
        
        for pos in positions:
            cached = self.cache.get(pos)
            if cached is not None:
                cache_hits[pos] = cached
            else:
                cache_misses.append(pos)
        
        # Add cache hits to results
        for pos in positions:
            if pos in cache_hits:
                results.append(cache_hits[pos])
            else:
                results.append(None)  # Placeholder
        
        # Compute missing values in batches
        if cache_misses:
            for i in range(0, len(cache_misses), batch_size):
                batch_positions = cache_misses[i:i + batch_size]
                
                # Compute batch
                keys, values = compute_fn(batch_positions)
                
                # Ensure tensors are batched
                if keys.dim() == 2:
                    # Single batch
                    keys = keys.unsqueeze(0)
                    values = values.unsqueeze(0)
                
                # Store in cache and update results
                for j, pos in enumerate(batch_positions):
                    key = keys[j] if keys.dim() > 1 else keys
                    value = values[j] if values.dim() > 1 else values
                    self.cache.put(pos, key, value)
                    
                    # Update result at original position
                    original_idx = positions.index(pos)
                    results[original_idx] = (key, value)
        
        return results
    
    def batch_update(
        self,
        updates: List[Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Batch update operation.
        
        Args:
            updates: List of (position, key, value) tuples (None = keep existing)
            batch_size: Batch size (None = use default)
            
        Returns:
            Number of entries updated
        """
        batch_size = batch_size or self.batch_size
        updated = 0
        
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            for pos, key, value in batch:
                cached = self.cache.get(pos)
                if cached is not None:
                    old_key, old_value = cached
                    new_key = key if key is not None else old_key
                    new_value = value if value is not None else old_value
                    self.cache.put(pos, new_key, new_value)
                    updated += 1
        
        return updated
    
    def batch_evict(
        self,
        positions: List[int],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Batch evict operation.
        
        Args:
            positions: List of positions to evict
            batch_size: Batch size (None = use default)
            
        Returns:
            Number of entries evicted
        """
        batch_size = batch_size or self.batch_size
        evicted = 0
        
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            evicted += self.cache.storage.remove(batch)
        
        return evicted
    
    def optimize_batch_size(
        self,
        sample_size: int = 100,
        test_batch_sizes: Optional[List[int]] = None
    ) -> int:
        """
        Optimize batch size based on performance testing.
        
        Args:
            sample_size: Number of operations to test
            test_batch_sizes: List of batch sizes to test (None = auto)
            
        Returns:
            Optimal batch size
        """
        import time
        
        if test_batch_sizes is None:
            test_batch_sizes = [8, 16, 32, 64, 128]
        
        best_batch_size = self.batch_size
        best_time = float('inf')
        
        # Create test positions
        positions = list(range(sample_size))
        
        for batch_size in test_batch_sizes:
            self.batch_size = batch_size
            
            # Time batch get
            start = time.time()
            self.batch_get(positions, batch_size=batch_size)
            elapsed = time.time() - start
            
            if elapsed < best_time:
                best_time = elapsed
                best_batch_size = batch_size
        
        self.batch_size = best_batch_size
        logger.info(f"Optimized batch size: {best_batch_size} (time: {best_time:.4f}s)")
        
        return best_batch_size

