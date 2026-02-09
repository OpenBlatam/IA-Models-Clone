"""
Enhanced Cache System for Flux2 Clothing Changer
=================================================

Advanced caching with intelligent eviction and prediction.
"""

import time
import hashlib
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedCache:
    """Enhanced caching system with intelligent eviction."""
    
    def __init__(
        self,
        max_size_mb: float = 1024.0,
        max_entries: int = 10000,
        ttl_seconds: Optional[float] = None,
        eviction_policy: str = "lru",  # lru, lfu, fifo, value_based
    ):
        """
        Initialize enhanced cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            ttl_seconds: Time to live in seconds
            eviction_policy: Eviction policy
        """
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """
        Calculate size of value in bytes.
        
        Args:
            value: Value to measure
            
        Returns:
            Size in bytes
        """
        import sys
        return sys.getsizeof(value)
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        if len(self.cache) >= self.max_entries:
            return True
        
        if self.total_size_bytes >= self.max_size_mb * 1024 * 1024:
            return True
        
        return False
    
    def _evict_entry(self) -> Optional[str]:
        """
        Evict entry based on policy.
        
        Returns:
            Evicted key
        """
        if not self.cache:
            return None
        
        if self.eviction_policy == "lru":
            # Least recently used
            key = next(iter(self.cache))
        elif self.eviction_policy == "lfu":
            # Least frequently used
            key = min(self.cache.items(), key=lambda x: x[1].access_count)[0]
        elif self.eviction_policy == "fifo":
            # First in first out
            key = next(iter(self.cache))
        elif self.eviction_policy == "value_based":
            # Largest value first
            key = max(self.cache.items(), key=lambda x: x[1].size_bytes)[0]
        else:
            key = next(iter(self.cache))
        
        entry = self.cache.pop(key)
        self.total_size_bytes -= entry.size_bytes
        self.evictions += 1
        
        logger.debug(f"Evicted cache entry: {key}")
        return key
    
    def _cleanup_expired(self) -> int:
        """
        Cleanup expired entries.
        
        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0
        
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.created_at > self.ttl_seconds
        ]
        
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.total_size_bytes -= entry.size_bytes
        
        return len(expired_keys)
    
    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Cleanup expired entries
        self._cleanup_expired()
        
        if key not in self.cache:
            self.misses += 1
            return default
        
        entry = self.cache[key]
        
        # Update access info
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        self.hits += 1
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
        """
        # Cleanup expired entries
        self._cleanup_expired()
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        # Check if we need to evict
        while self._should_evict():
            self._evict_entry()
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=size_bytes,
            metadata=metadata or {},
        )
        
        # Remove old entry if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.total_size_bytes -= old_entry.size_bytes
        
        # Add new entry
        self.cache[key] = entry
        self.total_size_bytes += size_bytes
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if key not in self.cache:
            return False
        
        entry = self.cache.pop(key)
        self.total_size_bytes -= entry.size_bytes
        return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.total_size_bytes = 0
        logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "size_mb": self.total_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_size_mb,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "eviction_policy": self.eviction_policy,
        }


