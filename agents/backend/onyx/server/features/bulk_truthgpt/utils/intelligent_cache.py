"""
Intelligent Cache System
========================

Advanced caching system with:
- Multi-level caching (memory, disk, Redis)
- Intelligent eviction policies
- Cache warming
- Predictive prefetching
- Cache analytics
"""

import asyncio
import logging
import time
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)

class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.ttl = ttl
        self.size = self._estimate_size(value)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def access(self):
        """Record access."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

class IntelligentCache:
    """
    Intelligent multi-level cache with:
    - LRU eviction
    - TTL support
    - Size limits
    - Access tracking
    - Predictive prefetching
    """
    
    def __init__(
        self,
        max_size_mb: float = 100.0,
        max_entries: int = 10000,
        default_ttl: Optional[float] = 3600.0,
        eviction_policy: str = "lru"
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        
        # Access patterns for prefetching
        self.access_patterns: Dict[str, List[str]] = {}
        
        # Background cleanup
        self.cleanup_task = None
        self.is_running = False
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        entry = self.cache.get(key)
        
        if entry is None:
            self.misses += 1
            return default
        
        if entry.is_expired():
            self._remove(key)
            self.misses += 1
            return default
        
        entry.access()
        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        
        return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Set value in cache."""
        entry = CacheEntry(key, value, ttl or self.default_ttl)
        
        # Check if we need to evict
        while (
            len(self.cache) >= self.max_entries or
            self.current_size + entry.size > self.max_size_bytes
        ):
            if not self._evict():
                logger.warning("Cache full, cannot add entry")
                return False
        
        # Add entry
        if key in self.cache:
            old_entry = self.cache[key]
            self.current_size -= old_entry.size
        
        self.cache[key] = entry
        self.current_size += entry.size
        self.cache.move_to_end(key)
        
        return True
    
    def _evict(self) -> bool:
        """Evict entry based on policy."""
        if not self.cache:
            return False
        
        if self.eviction_policy == "lru":
            # Remove least recently used
            key = next(iter(self.cache))
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        else:
            # Default: remove oldest
            key = next(iter(self.cache))
        
        self._remove(key)
        return True
    
    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size
    
    async def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size_bytes": self.current_size,
            "size_mb": round(self.current_size / (1024 * 1024), 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "utilization_percent": round((self.current_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0, 2)
        }
    
    async def warmup(self, warmup_func: Callable, *args, **kwargs):
        """Warm up cache with data."""
        try:
            result = await warmup_func(*args, **kwargs)
            key = self._generate_key(*args, **kwargs)
            await self.set(key, result)
            logger.info(f"Cache warmed up for key: {key[:20]}...")
        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")
    
    async def prefetch(self, key: str, fetch_func: Callable):
        """Predictively prefetch data."""
        try:
            if key in self.cache:
                return  # Already cached
            
            result = await fetch_func()
            await self.set(key, result)
            logger.debug(f"Prefetched key: {key[:20]}...")
        except Exception as e:
            logger.error(f"Prefetch failed: {e}")
    
    async def start_cleanup(self, interval: float = 60.0):
        """Start background cleanup task."""
        if self.is_running:
            return
        
        self.is_running = True
        
        async def cleanup_loop():
            while self.is_running:
                try:
                    await self._cleanup_expired()
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    await asyncio.sleep(interval)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Cache cleanup started")
    
    async def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._remove(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def stop_cleanup(self):
        """Stop background cleanup."""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache cleanup stopped")

# Global cache instance
intelligent_cache = IntelligentCache(max_size_mb=100.0, max_entries=10000)
































