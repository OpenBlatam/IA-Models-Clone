"""
Webhook Cache - Intelligent caching system for webhook data
"""

import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0
    tags: set = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1


class WebhookCache:
    """
    Intelligent caching system for webhook data
    
    Features:
    - TTL-based expiration
    - LRU eviction
    - Tag-based invalidation
    - Size-based eviction
    - Hit/miss statistics
    - Memory usage tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,
        max_memory_mb: float = 100.0
    ):
        """
        Initialize webhook cache
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "memory_usage_bytes": 0
        }
        
        # Tag index for fast invalidation
        self._tag_index: Dict[str, set] = defaultdict(set)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self.stats["misses"] += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry.is_expired():
            self._remove_entry(key)
            self.stats["expirations"] += 1
            self.stats["misses"] += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        
        self.stats["hits"] += 1
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[set] = None
    ) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            tags: Tags for invalidation
        """
        # Calculate expiration time
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl
        elif self.default_ttl > 0:
            expires_at = time.time() + self.default_ttl
        
        # Remove existing entry if present
        if key in self._cache:
            self._remove_entry(key)
        
        # Create new entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
            tags=tags or set()
        )
        
        # Add to cache
        self._cache[key] = entry
        
        # Update tag index
        for tag in entry.tags:
            self._tag_index[tag].add(key)
        
        # Update memory usage
        self._update_memory_usage()
        
        # Evict if necessary
        self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        if key not in self._cache:
            return False
        
        self._remove_entry(key)
        return True
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all entries with given tag
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if tag not in self._tag_index:
            return 0
        
        keys_to_remove = list(self._tag_index[tag])
        for key in keys_to_remove:
            self._remove_entry(key)
        
        return len(keys_to_remove)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern
        
        Args:
            pattern: Pattern to match (simple wildcard)
            
        Returns:
            Number of entries invalidated
        """
        import fnmatch
        
        keys_to_remove = [
            key for key in self._cache.keys()
            if fnmatch.fnmatch(key, pattern)
        ]
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._tag_index.clear()
        self.stats["memory_usage_bytes"] = 0
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        tags: Optional[set] = None
    ) -> Any:
        """
        Get value or set using factory function
        
        Args:
            key: Cache key
            factory: Function to generate value if not cached
            ttl: Time to live in seconds
            tags: Tags for invalidation
            
        Returns:
            Cached or generated value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        value = factory()
        self.set(key, value, ttl=ttl, tags=tags)
        return value
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update indexes"""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Remove from tag index
        for tag in entry.tags:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]
        
        # Remove from cache
        del self._cache[key]
        
        # Update memory usage
        self._update_memory_usage()
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full or memory limit exceeded"""
        # Check size limit
        while len(self._cache) > self.max_size:
            self._evict_lru()
        
        # Check memory limit
        while self.stats["memory_usage_bytes"] > self.max_memory_bytes:
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Remove oldest entry (first in OrderedDict)
        key, entry = self._cache.popitem(last=False)
        
        # Remove from tag index
        for tag in entry.tags:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]
        
        self.stats["evictions"] += 1
        self._update_memory_usage()
    
    def _update_memory_usage(self) -> None:
        """Update memory usage statistics"""
        total_size = 0
        
        for entry in self._cache.values():
            # Estimate size (rough calculation)
            key_size = len(entry.key.encode('utf-8'))
            value_size = self._estimate_value_size(entry.value)
            entry_size = key_size + value_size + 100  # Overhead
            total_size += entry_size
        
        self.stats["memory_usage_bytes"] = total_size
    
    def _estimate_value_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value).encode('utf-8'))
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 100  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "memory_usage_mb": self.stats["memory_usage_bytes"] / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024)
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats["expirations"] += 1
        
        return len(expired_keys)
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create deterministic string representation
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Create hash
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache instance
webhook_cache = WebhookCache()





