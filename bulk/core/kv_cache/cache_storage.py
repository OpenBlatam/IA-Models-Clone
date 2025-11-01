"""
Cache storage module for KV Cache.

Handles actual storage and retrieval of cached entries.
"""
import logging
import threading
import time
from typing import Dict, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class CacheStorage:
    """
    Thread-safe storage for cache entries.
    
    Responsibilities:
    - Store and retrieve cache entries
    - Track access times and counts
    - Thread-safe operations
    - Memory-efficient storage
    """
    
    def __init__(self, use_ordered_dict: bool = False):
        """Initialize cache storage with optimizations."""
        # Use OrderedDict for LRU-style fast access if needed
        if use_ordered_dict:
            self._cache: OrderedDict = OrderedDict()
        else:
            self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        self._access_times: Dict[int, float] = {}
        self._access_counts: Dict[int, int] = {}
        self._lock = threading.Lock()
        
        # Pre-allocate common sizes for faster operations
        self._fast_get = getattr(self, '_get_fast', None)
        
        logger.debug("Initialized CacheStorage with optimizations")
    
    def get(self, position: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached entry at position.
        
        Args:
            position: Cache position
            
        Returns:
            Tuple of (key, value) if found, None otherwise
        """
        with self._lock:
            if position in self._cache:
                # Update access tracking
                self._access_times[position] = time.time()
                self._access_counts[position] = self._access_counts.get(position, 0) + 1
                return self._cache[position]
            return None
    
    def put(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Store entry in cache.
        
        Args:
            position: Cache position
            key: Key tensor
            value: Value tensor
        """
        with self._lock:
            # Clone tensors to avoid reference issues
            self._cache[position] = (key.detach().clone(), value.detach().clone())
            self._access_times[position] = time.time()
            self._access_counts[position] = 1
    
    def remove(self, positions: list[int]) -> int:
        """
        Remove entries from cache.
        
        Args:
            positions: List of positions to remove
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            removed = 0
            for position in positions:
                if position in self._cache:
                    # Explicitly delete tensors
                    cached_key, cached_value = self._cache.pop(position)
                    del cached_key, cached_value
                    self._access_times.pop(position, None)
                    self._access_counts.pop(position, None)
                    removed += 1
            return removed
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Explicitly delete all tensors
            for cached_key, cached_value in self._cache.values():
                del cached_key, cached_value
            
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            return len(self._cache)
    
    def get_access_times(self) -> Dict[int, float]:
        """Get copy of access times."""
        with self._lock:
            return dict(self._access_times)
    
    def get_access_counts(self) -> Dict[int, int]:
        """Get copy of access counts."""
        with self._lock:
            return dict(self._access_counts)
    
    def contains(self, position: int) -> bool:
        """Check if position exists in cache."""
        with self._lock:
            return position in self._cache
    
    def get_positions(self) -> list[int]:
        """Get all cache positions."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_total_memory_mb(self) -> float:
        """Calculate total memory usage in MB."""
        with self._lock:
            total = 0
            for key, value in self._cache.values():
                total += key.numel() * key.element_size()
                total += value.numel() * value.element_size()
            return total / (1024**2)

