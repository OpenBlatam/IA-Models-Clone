"""
Advanced Caching System
Features: Redis distributed caching, cache invalidation patterns, performance optimization
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Caching imports
try:
    import aioredis
    from aioredis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcached
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"

class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Distributed cache (Redis)
    L3 = "l3"  # Database cache

@dataclass
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy = CacheStrategy.TTL
    default_ttl: int = 3600  # seconds
    max_size: int = 1000
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    key_prefix: str = "cache"
    namespace: str = "default"

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0

class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass

class RedisCacheBackend(CacheBackend):
    """Redis cache backend implementation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", config: CacheConfig = None):
        self.redis_url = redis_url
        self.config = config or CacheConfig()
        self.redis: Optional[Redis] = None
        self.stats = CacheStats()
    
    async def connect(self):
        """Connect to Redis"""
        try:
            if not REDIS_AVAILABLE:
                raise ImportError("aioredis is required for Redis cache backend")
            
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis cache backend")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        if self.config.serialization_format == "json":
            return json.dumps(value, default=str)
        elif self.config.serialization_format == "pickle":
            import pickle
            return pickle.dumps(value).hex()
        else:
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage"""
        if self.config.serialization_format == "json":
            return json.loads(value)
        elif self.config.serialization_format == "pickle":
            import pickle
            return pickle.loads(bytes.fromhex(value))
        else:
            return value
    
    def _get_key(self, key: str) -> str:
        """Get full cache key with namespace and prefix"""
        return f"{self.config.namespace}:{self.config.key_prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            full_key = self._get_key(key)
            value = await self.redis.get(full_key)
            
            if value:
                self.stats.hits += 1
                return self._deserialize(value.decode())
            else:
                self.stats.misses += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            self.stats.misses += 1
            return None
        finally:
            self.stats.total_requests += 1
            self._update_rates()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            full_key = self._get_key(key)
            serialized_value = self._serialize(value)
            ttl = ttl or self.config.default_ttl
            
            await self.redis.setex(full_key, ttl, serialized_value)
            self.stats.sets += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            full_key = self._get_key(key)
            result = await self.redis.delete(full_key)
            self.stats.deletes += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            full_key = self._get_key(key)
            return bool(await self.redis.exists(full_key))
            
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache keys in namespace"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            pattern = f"{self.config.namespace}:{self.config.key_prefix}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats
    
    def _update_rates(self):
        """Update hit/miss rates"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests
            self.stats.miss_rate = self.stats.misses / self.stats.total_requests

class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend implementation"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache"""
        async with self._lock:
            try:
                if key in self.cache:
                    cache_entry = self.cache[key]
                    
                    # Check TTL
                    if time.time() > cache_entry["expires_at"]:
                        del self.cache[key]
                        self.access_times.pop(key, None)
                        self.access_counts.pop(key, None)
                        self.stats.evictions += 1
                        self.stats.misses += 1
                        return None
                    
                    # Update access tracking
                    self.access_times[key] = time.time()
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    
                    self.stats.hits += 1
                    return cache_entry["value"]
                else:
                    self.stats.misses += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to get cache key {key}: {e}")
                self.stats.misses += 1
                return None
            finally:
                self.stats.total_requests += 1
                self._update_rates()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache"""
        async with self._lock:
            try:
                ttl = ttl or self.config.default_ttl
                expires_at = time.time() + ttl
                
                # Check cache size limit
                if len(self.cache) >= self.config.max_size and key not in self.cache:
                    await self._evict_entries()
                
                self.cache[key] = {
                    "value": value,
                    "expires_at": expires_at,
                    "created_at": time.time()
                }
                
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                
                self.stats.sets += 1
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from in-memory cache"""
        async with self._lock:
            try:
                if key in self.cache:
                    del self.cache[key]
                    self.access_times.pop(key, None)
                    self.access_counts.pop(key, None)
                    self.stats.deletes += 1
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Failed to delete cache key {key}: {e}")
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache"""
        async with self._lock:
            if key in self.cache:
                cache_entry = self.cache[key]
                if time.time() <= cache_entry["expires_at"]:
                    return True
                else:
                    # Expired, remove it
                    del self.cache[key]
                    self.access_times.pop(key, None)
                    self.access_counts.pop(key, None)
                    self.stats.evictions += 1
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        async with self._lock:
            try:
                self.cache.clear()
                self.access_times.clear()
                self.access_counts.clear()
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats
    
    async def _evict_entries(self):
        """Evict entries based on strategy"""
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self.access_times:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                self.access_counts.pop(oldest_key, None)
                self.stats.evictions += 1
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self.access_counts:
                least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
                del self.cache[least_used_key]
                del self.access_counts[least_used_key]
                self.access_times.pop(least_used_key, None)
                self.stats.evictions += 1
    
    def _update_rates(self):
        """Update hit/miss rates"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests
            self.stats.miss_rate = self.stats.misses / self.stats.total_requests

class CacheManager:
    """
    Advanced cache manager with multiple backends and strategies
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.backends: Dict[CacheLevel, CacheBackend] = {}
        self.stats = CacheStats()
    
    def add_backend(self, level: CacheLevel, backend: CacheBackend):
        """Add cache backend"""
        self.backends[level] = backend
    
    async def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """Get value from cache with fallback strategy"""
        if level:
            # Get from specific level
            if level in self.backends:
                return await self.backends[level].get(key)
            return None
        
        # Multi-level cache lookup
        for cache_level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
            if cache_level in self.backends:
                value = await self.backends[cache_level].get(key)
                if value is not None:
                    # Populate higher levels with the value
                    for higher_level in [CacheLevel.L1, CacheLevel.L2]:
                        if higher_level in self.backends and higher_level.value < cache_level.value:
                            await self.backends[higher_level].set(key, value)
                    return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, level: Optional[CacheLevel] = None) -> bool:
        """Set value in cache"""
        if level:
            # Set in specific level
            if level in self.backends:
                return await self.backends[level].set(key, value, ttl)
            return False
        
        # Set in all levels
        success = True
        for cache_level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
            if cache_level in self.backends:
                result = await self.backends[cache_level].set(key, value, ttl)
                success = success and result
        
        return success
    
    async def delete(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """Delete value from cache"""
        if level:
            # Delete from specific level
            if level in self.backends:
                return await self.backends[level].delete(key)
            return False
        
        # Delete from all levels
        success = True
        for backend in self.backends.values():
            result = await backend.delete(key)
            success = success and result
        
        return success
    
    async def exists(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """Check if key exists in cache"""
        if level:
            # Check specific level
            if level in self.backends:
                return await self.backends[level].exists(key)
            return False
        
        # Check any level
        for backend in self.backends.values():
            if await backend.exists(key):
                return True
        
        return False
    
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """Clear cache"""
        if level:
            # Clear specific level
            if level in self.backends:
                return await self.backends[level].clear()
            return False
        
        # Clear all levels
        success = True
        for backend in self.backends.values():
            result = await backend.clear()
            success = success and result
        
        return success
    
    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all backends"""
        stats = {}
        for level, backend in self.backends.items():
            stats[level.value] = await backend.get_stats()
        return stats

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()

def cached(ttl: int = 3600, key_func: Optional[Callable] = None, cache_manager: Optional[CacheManager] = None):
    """Decorator for caching function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Get cache manager
            manager = cache_manager or get_default_cache_manager()
            
            # Try to get from cache
            cached_result = await manager.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await manager.set(cache_key_str, result, ttl)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Get cache manager
            manager = cache_manager or get_default_cache_manager()
            
            # Try to get from cache (sync)
            import asyncio
            loop = asyncio.get_event_loop()
            cached_result = loop.run_until_complete(manager.get(cache_key_str))
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result (sync)
            loop.run_until_complete(manager.set(cache_key_str, result, ttl))
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global cache manager instance
_default_cache_manager: Optional[CacheManager] = None

def get_default_cache_manager() -> CacheManager:
    """Get default cache manager instance"""
    global _default_cache_manager
    if not _default_cache_manager:
        _default_cache_manager = CacheManager()
        
        # Add in-memory backend
        l1_config = CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=1000,
            default_ttl=300
        )
        _default_cache_manager.add_backend(CacheLevel.L1, InMemoryCacheBackend(l1_config))
        
        # Add Redis backend if available
        if REDIS_AVAILABLE:
            l2_config = CacheConfig(
                strategy=CacheStrategy.TTL,
                default_ttl=3600,
                namespace="microservices"
            )
            _default_cache_manager.add_backend(CacheLevel.L2, RedisCacheBackend(config=l2_config))
    
    return _default_cache_manager

async def initialize_cache_manager(redis_url: str = "redis://localhost:6379"):
    """Initialize cache manager with Redis backend"""
    global _default_cache_manager
    
    _default_cache_manager = CacheManager()
    
    # Add in-memory backend
    l1_config = CacheConfig(
        strategy=CacheStrategy.LRU,
        max_size=1000,
        default_ttl=300
    )
    _default_cache_manager.add_backend(CacheLevel.L1, InMemoryCacheBackend(l1_config))
    
    # Add Redis backend
    if REDIS_AVAILABLE:
        l2_config = CacheConfig(
            strategy=CacheStrategy.TTL,
            default_ttl=3600,
            namespace="microservices"
        )
        redis_backend = RedisCacheBackend(redis_url, l2_config)
        await redis_backend.connect()
        _default_cache_manager.add_backend(CacheLevel.L2, redis_backend)
    
    return _default_cache_manager






























