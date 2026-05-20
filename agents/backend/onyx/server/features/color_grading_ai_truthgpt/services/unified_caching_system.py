"""
Unified Caching System for Color Grading AI
============================================

Consolidated caching system combining:
- UnifiedCache (local + Redis + disk)
- AdvancedCache (multiple strategies)
- CachingStrategy (decorator support)

Features:
- Multi-tier caching (memory, Redis, disk)
- Multiple eviction strategies (LRU, LFU, FIFO, TTL, Adaptive)
- Cache warming and invalidation
- Statistics and monitoring
- Async support
- Decorator support
"""

import logging
import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedCachingSystem:
    """
    Unified caching system combining all caching capabilities.
    
    Features:
    - Multi-tier: Memory, Redis (optional), Disk
    - Multiple eviction strategies
    - Cache warming and invalidation
    - Statistics tracking
    - Async support
    - Decorator support
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize unified caching system.
        
        Args:
            cache_dir: Directory for disk cache
            max_size: Maximum memory cache size
            strategy: Cache eviction strategy
            default_ttl: Default TTL in seconds
            redis_url: Optional Redis connection URL
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = timedelta(seconds=default_ttl) if default_ttl else None
        self.redis_url = redis_url
        
        # Memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        
        # Redis client (optional)
        self._redis_client = None
        if redis_url:
            self._init_redis()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
        }
    
    def _init_redis(self):
        """Initialize Redis client."""
        try:
            import redis
            self._redis_client = redis.from_url(self.redis_url)
            self._redis_client.ping()
            logger.info("Connected to Redis for distributed cache")
        except ImportError:
            logger.warning("Redis not installed. Using local cache only.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using local cache.")
            self._redis_client = None
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (tries memory, Redis, disk).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self._lock:
            # Try memory cache first
            entry = self._cache.get(key)
            if entry:
                # Check TTL
                if entry.ttl and (datetime.now() - entry.created_at) > entry.ttl:
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
                
                # Update access
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._stats["hits"] += 1
                return entry.value
            
            # Try Redis
            if self._redis_client:
                try:
                    value = self._redis_client.get(key)
                    if value:
                        # Promote to memory
                        await self._set_memory(key, json.loads(value))
                        self._stats["redis_hits"] += 1
                        self._stats["hits"] += 1
                        return json.loads(value)
                    self._stats["redis_misses"] += 1
                except Exception as e:
                    logger.debug(f"Redis get error: {e}")
            
            # Try disk cache
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        entry_data = json.load(f)
                    
                    expires_at = datetime.fromisoformat(entry_data.get("expires_at", "9999-12-31"))
                    if expires_at > datetime.now():
                        value = entry_data["value"]
                        # Promote to memory
                        await self._set_memory(key, value)
                        self._stats["disk_hits"] += 1
                        self._stats["hits"] += 1
                        return value
                    else:
                        cache_file.unlink()  # Expired
                except Exception as e:
                    logger.debug(f"Error reading cache file: {e}")
            
            self._stats["disk_misses"] += 1
            self._stats["misses"] += 1
            return None
    
    async def _set_memory(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Set value in memory cache."""
        # Check if we need to evict
        if len(self._cache) >= self.max_size and key not in self._cache:
            await self._evict()
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl or self.default_ttl
        )
        self._cache[key] = entry
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache (all tiers).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            metadata: Optional metadata
        """
        async with self._lock:
            ttl_delta = timedelta(seconds=ttl) if ttl else self.default_ttl
            
            # Set in memory
            await self._set_memory(key, value, ttl_delta)
            if metadata:
                self._cache[key].metadata = metadata
            
            # Set in Redis
            if self._redis_client:
                try:
                    expires_at = datetime.now() + ttl_delta if ttl_delta else None
                    ttl_seconds = int(ttl_delta.total_seconds()) if ttl_delta else None
                    self._redis_client.setex(
                        key,
                        ttl_seconds or 0,
                        json.dumps(value, default=str)
                    )
                except Exception as e:
                    logger.debug(f"Redis set error: {e}")
            
            # Set in disk cache
            try:
                cache_file = self.cache_dir / f"{key}.json"
                expires_at = datetime.now() + ttl_delta if ttl_delta else datetime.max
                with open(cache_file, "w") as f:
                    json.dump({
                        "value": value,
                        "expires_at": expires_at.isoformat(),
                        "metadata": metadata or {}
                    }, f, default=str)
            except Exception as e:
                logger.debug(f"Error writing cache file: {e}")
            
            self._stats["sets"] += 1
    
    async def _evict(self):
        """Evict entry based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )[0]
        
        elif self.strategy == CacheStrategy.LFU:
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )[0]
        
        elif self.strategy == CacheStrategy.FIFO:
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )[0]
        
        elif self.strategy == CacheStrategy.TTL:
            now = datetime.now()
            expired = [
                (k, v) for k, v in self._cache.items()
                if v.ttl and (now - v.created_at) > v.ttl
            ]
            if expired:
                key_to_remove = expired[0][0]
            else:
                key_to_remove = min(
                    self._cache.items(),
                    key=lambda x: x[1].created_at
                )[0]
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            scores = {
                k: v.access_count / max((datetime.now() - v.last_accessed).total_seconds(), 1)
                for k, v in self._cache.items()
            }
            key_to_remove = min(scores.items(), key=lambda x: x[1])[0]
        
        else:
            key_to_remove = next(iter(self._cache))
        
        del self._cache[key_to_remove]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache key: {key_to_remove}")
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        async with self._lock:
            removed = False
            
            if key in self._cache:
                del self._cache[key]
                removed = True
            
            if self._redis_client:
                try:
                    self._redis_client.delete(key)
                except Exception as e:
                    logger.debug(f"Redis delete error: {e}")
            
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()
                removed = True
            
            return removed
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern."""
        async with self._lock:
            count = 0
            
            # Memory
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
            
            # Redis
            if self._redis_client:
                try:
                    keys = self._redis_client.keys(f"*{pattern}*")
                    if keys:
                        self._redis_client.delete(*keys)
                        count += len(keys)
                except Exception as e:
                    logger.debug(f"Redis pattern delete error: {e}")
            
            # Disk
            for cache_file in self.cache_dir.glob(f"*{pattern}*.json"):
                cache_file.unlink()
                count += 1
            
            return count
    
    async def clear(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
            
            if self._redis_client:
                try:
                    self._redis_client.flushdb()
                except Exception as e:
                    logger.debug(f"Redis clear error: {e}")
            
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            logger.info("Cache cleared")
    
    async def warm_cache(self, items: Dict[str, Any]):
        """Warm cache with items."""
        for key, value in items.items():
            await self.set(key, value)
        logger.info(f"Warmed cache with {len(items)} items")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "sets": self._stats["sets"],
            "redis_hits": self._stats["redis_hits"],
            "redis_misses": self._stats["redis_misses"],
            "disk_hits": self._stats["disk_hits"],
            "disk_misses": self._stats["disk_misses"],
            "disk_cache_files": len(list(self.cache_dir.glob("*.json"))),
        }
    
    def cache_decorator(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Optional TTL in seconds
            key_func: Optional key generation function
        """
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(*args, **kwargs)
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator


