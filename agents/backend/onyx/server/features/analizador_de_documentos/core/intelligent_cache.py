"""
Intelligent Cache for Document Analyzer
========================================

Advanced caching system with predictive prefetching, cache warming, and intelligent eviction.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class IntelligentCache:
    """Intelligent cache with advanced features"""
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        enable_prefetch: bool = True
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.enable_prefetch = enable_prefetch
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()
        self.access_frequency: Dict[str, int] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(
            f"IntelligentCache initialized. "
            f"Max size: {max_size}, Policy: {eviction_policy.value}, TTL: {default_ttl}"
        )
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
        
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            for k, v in sorted_kwargs:
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:16]}")
        
        return ":".join(key_parts)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        entry = self.cache.get(key)
        
        if entry is None:
            self.misses += 1
            return default
        
        # Check expiration
        if entry.is_expired():
            self.delete(key)
            self.misses += 1
            return default
        
        # Update access metadata
        entry.touch()
        self.access_order.move_to_end(key)
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
        
        self.hits += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache"""
        # Check if we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        # Create entry
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl=ttl_to_use,
            size=self._estimate_size(value)
        )
        
        self.cache[key] = entry
        self.access_order[key] = time.time()
        self.access_frequency[key] = 1
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]
            if key in self.access_frequency:
                del self.access_frequency[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()
        self.access_frequency.clear()
        logger.info("Cache cleared")
    
    def _evict(self):
        """Evict entry based on policy"""
        if not self.cache:
            return
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove least recently used
            lru_key = next(iter(self.access_order))
            self.delete(lru_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            lfu_key = min(self.access_frequency, key=self.access_frequency.get)
            self.delete(lfu_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # Remove oldest
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            self.delete(oldest_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Remove expired first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                self.delete(expired_keys[0])
            else:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                self.delete(oldest_key)
        
        self.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            return 0
    
    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable,
        ttl: Optional[float] = None,
        *args,
        **kwargs
    ) -> Any:
        """Get from cache or fetch using function"""
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Fetch new value
        if asyncio.iscoroutinefunction(fetch_func):
            value = await fetch_func(*args, **kwargs)
        else:
            value = fetch_func(*args, **kwargs)
        
        # Store in cache
        self.set(key, value, ttl=ttl)
        
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "eviction_policy": self.eviction_policy.value
        }
    
    def cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")

# Global instance
intelligent_cache = IntelligentCache()
















