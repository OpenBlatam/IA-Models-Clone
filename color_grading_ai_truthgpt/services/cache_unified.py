"""
Unified Cache for Color Grading AI
===================================

Unified cache implementation with local and distributed support.
Enhanced with advanced caching strategies from AdvancedCache.
"""

import logging
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage


class UnifiedCache:
    """
    Unified cache with local and distributed (Redis) support.
    
    Features:
    - Local cache with TTL
    - Redis backend (optional)
    - Automatic fallback
    - Cache invalidation
    - Statistics
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        ttl: int = 3600,
        redis_url: Optional[str] = None,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Initialize unified cache.
        
        Args:
            cache_dir: Directory for local cache
            ttl: Default TTL in seconds
            redis_url: Optional Redis connection URL
            max_size: Maximum cache size (for local memory cache)
            strategy: Cache eviction strategy
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.redis_url = redis_url
        self.max_size = max_size
        self.strategy = strategy
        self._redis_client = None
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
        }
        
        if redis_url:
            self._init_redis()
    
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
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try Redis first
        if self._redis_client:
            try:
                value = self._redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # Try local memory cache
        async with self._lock:
            if key in self._local_cache:
                entry = self._local_cache[key]
                if entry["expires_at"] > datetime.now():
                    # Update access tracking for LRU/LFU
                    entry["last_accessed"] = datetime.now()
                    entry["access_count"] = entry.get("access_count", 0) + 1
                    self._stats["hits"] += 1
                    return entry["value"]
                else:
                    del self._local_cache[key]
                    self._stats["misses"] += 1
            else:
                self._stats["misses"] += 1
        
        # Try disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    entry = json.load(f)
                
                expires_at = datetime.fromisoformat(entry["expires_at"])
                if expires_at > datetime.now():
                    return entry["value"]
                else:
                    cache_file.unlink()  # Expired
            except Exception as e:
                logger.debug(f"Error reading cache file: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        ttl = ttl or self.ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        async with self._lock:
            # Check if we need to evict
            if len(self._local_cache) >= self.max_size and key not in self._local_cache:
                await self._evict()
            
            # Set in Redis
            if self._redis_client:
                try:
                    self._redis_client.setex(
                        key,
                        ttl,
                        json.dumps(value, default=str)
                    )
                except Exception as e:
                    logger.debug(f"Redis set error: {e}")
            
            # Set in local memory cache
            self._local_cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 0
            }
            self._stats["sets"] += 1
        
        # Set in disk cache
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, "w") as f:
                json.dump({
                    "value": value,
                    "expires_at": expires_at.isoformat()
                }, f, default=str)
        except Exception as e:
            logger.debug(f"Error writing cache file: {e}")
    
    async def _evict(self):
        """Evict entry based on strategy."""
        if not self._local_cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key_to_remove = min(
                self._local_cache.items(),
                key=lambda x: x[1].get("last_accessed", datetime.min)
            )[0]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key_to_remove = min(
                self._local_cache.items(),
                key=lambda x: x[1].get("access_count", 0)
            )[0]
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest
            key_to_remove = min(
                self._local_cache.items(),
                key=lambda x: x[1].get("created_at", datetime.min)
            )[0]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired or oldest
            now = datetime.now()
            expired = [
                (k, v) for k, v in self._local_cache.items()
                if v.get("expires_at", datetime.max) < now
            ]
            if expired:
                key_to_remove = expired[0][0]
            else:
                key_to_remove = min(
                    self._local_cache.items(),
                    key=lambda x: x[1].get("created_at", datetime.min)
                )[0]
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: combine LRU and LFU
            scores = {
                k: v.get("access_count", 0) / max(
                    (datetime.now() - v.get("last_accessed", datetime.now())).total_seconds(),
                    1
                )
                for k, v in self._local_cache.items()
            }
            key_to_remove = min(scores.items(), key=lambda x: x[1])[0]
        
        else:
            # Default: remove first
            key_to_remove = next(iter(self._local_cache))
        
        del self._local_cache[key_to_remove]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache key: {key_to_remove}")
    
    async def delete(self, key: str):
        """Delete key from cache."""
        # Delete from Redis
        if self._redis_client:
            try:
                self._redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
        
        # Delete from local cache
        if key in self._local_cache:
            del self._local_cache[key]
        
        # Delete from disk
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    async def clear(self, pattern: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            pattern: Optional key pattern
        """
        # Clear Redis
        if self._redis_client and pattern:
            try:
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis clear error: {e}")
        
        # Clear local cache
        if pattern:
            keys_to_delete = [k for k in self._local_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._local_cache[key]
        else:
            self._local_cache.clear()
        
        # Clear disk cache
        if pattern:
            for cache_file in self.cache_dir.glob(f"*{pattern}*.json"):
                cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        stats = {
            "backend": "redis" if self._redis_client else "local",
            "strategy": self.strategy.value,
            "max_size": self.max_size,
            "local_cache_size": len(self._local_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.json"))),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "sets": self._stats["sets"],
        }
        
        if self._redis_client:
            try:
                info = self._redis_client.info("stats")
                stats["redis"] = {
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            except:
                pass
        
        return stats
    
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
                    cache_key = self._generate_key(func.__name__, *args, **kwargs)
                
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



