from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import pickle
from datetime import datetime, timedelta
from .config import get_config
from .optimization import get_optimization_manager
                import redis
                import redis
from typing import Any, List, Dict, Optional
"""
Cache Manager
=============

Multi-level caching system with memory, Redis, compression, and intelligent eviction.
"""



# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    compression_ratio: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        
    """__init__ function."""
self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self.stats = CacheStats()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl
    
    def _evict_expired(self) -> Any:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self._remove_key(key)
            self.stats.evictions += 1
    
    def _remove_key(self, key: str):
        """Remove key from cache and timestamps"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def _evict_lru(self) -> Any:
        """Evict least recently used items"""
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove_key(oldest_key)
            self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._evict_expired()
        
        if key not in self._cache or self._is_expired(key):
            self.stats.misses += 1
            return None
        
        # Move to end (most recently used)
        value = self._cache.pop(key)
        self._cache[key] = value
        self.stats.hits += 1
        return value
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self._evict_expired()
        self._evict_lru()
        
        # Remove if exists (to update position)
        if key in self._cache:
            del self._cache[key]
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self.stats.sets += 1
        
        # Update memory usage estimate
        try:
            self.stats.memory_usage = sum(
                len(pickle.dumps(v)) for v in self._cache.values()
            )
        except Exception:
            pass  # Ignore pickle errors for memory estimation
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self._cache:
            self._remove_key(key)
            self.stats.deletes += 1
            return True
        return False
    
    def clear(self) -> Any:
        """Clear all cache entries"""
        self._cache.clear()
        self._timestamps.clear()
        self.stats = CacheStats()
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        self._evict_expired()
        return list(self._cache.keys())


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self) -> Any:
        self.redis_client = None
        self.stats = CacheStats()
        self._setup_redis()
    
    def _setup_redis(self) -> Any:
        """Setup Redis connection"""
        try:
            config = get_config()
            
            # Try to import redis with hiredis for better performance
            optimization_manager = get_optimization_manager()
            if optimization_manager.profile.libraries.get("hiredis", None) and optimization_manager.profile.libraries["hiredis"].available:
                self.redis_client = redis.from_url(
                    config.redis.url,
                    max_connections=config.redis.max_connections,
                    socket_timeout=config.redis.socket_timeout,
                    socket_connect_timeout=config.redis.socket_connect_timeout,
                    retry_on_timeout=config.redis.retry_on_timeout,
                    parser_class=redis.connection.HiredisParser
                )
                logger.info("✓ Redis connected with hiredis parser")
            else:
                self.redis_client = redis.from_url(
                    config.redis.url,
                    max_connections=config.redis.max_connections,
                    socket_timeout=config.redis.socket_timeout,
                    socket_connect_timeout=config.redis.socket_connect_timeout,
                    retry_on_timeout=config.redis.retry_on_timeout
                )
                logger.info("✓ Redis connected with default parser")
                
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _get_key(self, key: str) -> str:
        """Get prefixed Redis key"""
        config = get_config()
        return f"{config.cache.redis_cache_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            self.stats.misses += 1
            return None
        
        try:
            redis_key = self._get_key(key)
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, redis_key
            )
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize data
            value = pickle.loads(data)
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache"""
        if not self.redis_client:
            return
        
        try:
            config = get_config()
            redis_key = self._get_key(key)
            ttl = ttl or config.cache.redis_cache_ttl
            
            # Serialize data
            data = pickle.dumps(value)
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.setex, redis_key, ttl, data
            )
            
            self.stats.sets += 1
            
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._get_key(key)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, redis_key
            )
            
            if result:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        if not self.redis_client:
            return
        
        try:
            redis_pattern = self._get_key(pattern)
            keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, redis_pattern
            )
            
            if keys:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.delete, *keys
                )
                self.stats.deletes += len(keys)
                
        except Exception as e:
            logger.warning(f"Redis clear pattern error: {e}")


class CacheManager:
    """Multi-level cache manager with compression and intelligent strategies"""
    
    def __init__(self) -> Any:
        config = get_config()
        self.optimization_manager = get_optimization_manager()
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(
            max_size=config.cache.memory_cache_size,
            ttl=config.cache.memory_cache_ttl
        )
        self.redis_cache = RedisCache()
        
        # Get optimized serializer and compressor
        self.serializer = self.optimization_manager.get_serializer()
        self.compressor = self.optimization_manager.get_compressor()
        self.hasher = self.optimization_manager.get_hasher()
        
        # Configuration
        self.enable_compression = config.cache.enable_compression
        self.compression_threshold = config.cache.compression_threshold
        
        logger.info(f"✓ Cache manager initialized with {self.serializer['name']} serializer "
                   f"and {self.compressor['name']} compressor")
    
    def _generate_cache_key(self, key: str, **kwargs) -> str:
        """Generate consistent cache key"""
        if kwargs:
            # Include kwargs in key for parameter-specific caching
            key_data = f"{key}:{self.serializer['dumps'](sorted(kwargs.items()))}"
        else:
            key_data = key
        
        return self.hasher(key_data)
    
    def _should_compress(self, data: bytes) -> bool:
        """Determine if data should be compressed"""
        return self.enable_compression and len(data) > self.compression_threshold
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress and serialize data"""
        # Serialize first
        serialized = self.serializer["dumps"](data).encode()
        
        # Compress if beneficial
        if self._should_compress(serialized):
            try:
                compressed = self.compressor["compress"](serialized)
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    return b"COMPRESSED:" + compressed
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
        
        return b"RAW:" + serialized
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress and deserialize data"""
        if data.startswith(b"COMPRESSED:"):
            try:
                compressed_data = data[11:]  # Remove "COMPRESSED:" prefix
                decompressed = self.compressor["decompress"](compressed_data)
                return self.serializer["loads"](decompressed.decode())
            except Exception as e:
                logger.warning(f"Decompression failed: {e}")
                raise
        elif data.startswith(b"RAW:"):
            raw_data = data[4:]  # Remove "RAW:" prefix
            return self.serializer["loads"](raw_data.decode())
        else:
            # Legacy format fallback
            return self.serializer["loads"](data.decode())
    
    async def get(self, key: str, **kwargs) -> Optional[Any]:
        """Get value from cache (L1: Memory, L2: Redis)"""
        cache_key = self._generate_cache_key(key, **kwargs)
        
        # L1: Check memory cache first
        value = self.memory_cache.get(cache_key)
        if value is not None:
            return value
        
        # L2: Check Redis cache
        redis_data = await self.redis_cache.get(cache_key)
        if redis_data is not None:
            try:
                # Decompress and deserialize
                if isinstance(redis_data, bytes):
                    value = self._decompress_data(redis_data)
                else:
                    value = redis_data
                
                # Store in memory cache for faster access
                self.memory_cache.set(cache_key, value)
                return value
            except Exception as e:
                logger.warning(f"Cache deserialization error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
        """Set value in cache"""
        cache_key = self._generate_cache_key(key, **kwargs)
        
        # Store in memory cache
        self.memory_cache.set(cache_key, value)
        
        # Store in Redis cache with compression
        try:
            compressed_data = self._compress_data(value)
            await self.redis_cache.set(cache_key, compressed_data, ttl)
        except Exception as e:
            logger.warning(f"Cache compression/storage error: {e}")
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete key from all cache layers"""
        cache_key = self._generate_cache_key(key, **kwargs)
        
        # Delete from memory cache
        memory_deleted = self.memory_cache.delete(cache_key)
        
        # Delete from Redis cache
        redis_deleted = await self.redis_cache.delete(cache_key)
        
        return memory_deleted or redis_deleted
    
    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern from all caches"""
        # Clear from memory cache
        memory_keys = self.memory_cache.keys()
        for key in memory_keys:
            if pattern in key:
                self.memory_cache.delete(key)
        
        # Clear from Redis cache
        await self.redis_cache.clear_pattern(f"*{pattern}*")
    
    async def clear_all(self) -> Any:
        """Clear all caches"""
        self.memory_cache.clear()
        await self.redis_cache.clear_pattern("*")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.stats
        redis_stats = self.redis_cache.stats
        
        total_hits = memory_stats.hits + redis_stats.hits
        total_misses = memory_stats.misses + redis_stats.misses
        total_requests = total_hits + total_misses
        
        return {
            "overall": {
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0.0
            },
            "memory_cache": {
                "hits": memory_stats.hits,
                "misses": memory_stats.misses,
                "sets": memory_stats.sets,
                "deletes": memory_stats.deletes,
                "evictions": memory_stats.evictions,
                "hit_rate": memory_stats.hit_rate,
                "memory_usage_bytes": memory_stats.memory_usage,
                "current_size": len(self.memory_cache._cache),
                "max_size": self.memory_cache.max_size
            },
            "redis_cache": {
                "hits": redis_stats.hits,
                "misses": redis_stats.misses,
                "sets": redis_stats.sets,
                "deletes": redis_stats.deletes,
                "hit_rate": redis_stats.hit_rate,
                "available": self.redis_cache.redis_client is not None
            },
            "optimization": {
                "serializer": self.serializer["name"],
                "compressor": self.compressor["name"],
                "compression_enabled": self.enable_compression,
                "compression_threshold": self.compression_threshold
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems"""
        health = {
            "memory_cache": {"status": "healthy", "details": {}},
            "redis_cache": {"status": "unknown", "details": {}}
        }
        
        # Test memory cache
        try:
            test_key = f"health_check_{int(time.time())}"
            self.memory_cache.set(test_key, "test_value")
            retrieved = self.memory_cache.get(test_key)
            self.memory_cache.delete(test_key)
            
            if retrieved == "test_value":
                health["memory_cache"]["status"] = "healthy"
            else:
                health["memory_cache"]["status"] = "unhealthy"
                health["memory_cache"]["details"]["error"] = "Value mismatch"
                
        except Exception as e:
            health["memory_cache"]["status"] = "unhealthy"
            health["memory_cache"]["details"]["error"] = str(e)
        
        # Test Redis cache
        try:
            if self.redis_cache.redis_client:
                test_key = f"health_check_{int(time.time())}"
                await self.redis_cache.set(test_key, "test_value", ttl=60)
                retrieved = await self.redis_cache.get(test_key)
                await self.redis_cache.delete(test_key)
                
                if retrieved == "test_value":
                    health["redis_cache"]["status"] = "healthy"
                else:
                    health["redis_cache"]["status"] = "unhealthy"
                    health["redis_cache"]["details"]["error"] = "Value mismatch"
            else:
                health["redis_cache"]["status"] = "unavailable"
                health["redis_cache"]["details"]["error"] = "Redis client not initialized"
                
        except Exception as e:
            health["redis_cache"]["status"] = "unhealthy"
            health["redis_cache"]["details"]["error"] = str(e)
        
        return health


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager"""
    return cache_manager 