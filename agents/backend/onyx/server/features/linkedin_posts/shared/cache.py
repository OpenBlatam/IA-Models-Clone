from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
from typing import Optional, Any, Dict, List, Callable
import orjson
import hashlib
from datetime import datetime, timedelta
import time
from functools import lru_cache, wraps
import redis.asyncio as redis
from cachetools import TTLCache, LFUCache
import pickle
import zlib
from dataclasses import dataclass
from enum import Enum
import logging
from .config import settings
from .logging import get_logger
        import fnmatch
from typing import Any, List, Dict, Optional
"""
Advanced Multi-Level Cache System
=================================

High-performance caching with memory, Redis, and distributed support.
"""



logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-tier caching."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISTRIBUTED = "l3_distributed"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """
    Advanced multi-level cache manager with automatic tiering.
    """
    
    def __init__(
        self,
        memory_size: int = 10000,
        memory_ttl: int = 300,
        redis_url: Optional[str] = None,
        compression_threshold: int = 1024,
        enable_stats: bool = True
    ):
        
    """__init__ function."""
# L1: Memory cache (fastest)
        self.memory_cache = TTLCache(maxsize=memory_size, ttl=memory_ttl)
        self.lfu_cache = LFUCache(maxsize=memory_size // 2)  # For hot data
        
        # L2: Redis cache (fast, persistent)
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_pool = None
        self._redis_client = None
        
        # Configuration
        self.compression_threshold = compression_threshold
        self.enable_stats = enable_stats
        
        # Statistics
        self.stats = {
            CacheLevel.L1_MEMORY: CacheStats(),
            CacheLevel.L2_REDIS: CacheStats(),
            CacheLevel.L3_DISTRIBUTED: CacheStats()
        }
        
        # Cache warming
        self._warm_cache_tasks = []
        
        # Invalidation callbacks
        self._invalidation_callbacks = []
    
    async def initialize(self) -> Any:
        """Initialize cache connections."""
        try:
            # Initialize Redis connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 3,  # TCP_KEEPCNT
                }
            )
            self._redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self._redis_client.ping()
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            # Continue without Redis (memory-only mode)
            self._redis_client = None
    
    async def get(
        self,
        key: str,
        default: Any = None,
        deserializer: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get value from cache with automatic tiering.
        """
        # Normalize key
        normalized_key = self._normalize_key(key)
        
        # L1: Check memory cache
        value = self._get_from_memory(normalized_key)
        if value is not None:
            self._record_hit(CacheLevel.L1_MEMORY)
            return value
        
        # L2: Check Redis cache
        if self._redis_client:
            value = await self._get_from_redis(normalized_key)
            if value is not None:
                self._record_hit(CacheLevel.L2_REDIS)
                
                # Promote to L1
                self._set_in_memory(normalized_key, value)
                
                return self._deserialize(value, deserializer)
        
        # Cache miss
        self._record_miss(CacheLevel.L1_MEMORY)
        if self._redis_client:
            self._record_miss(CacheLevel.L2_REDIS)
        
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        serializer: Optional[Callable] = None,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """
        Set value in cache with automatic tiering.
        """
        # Normalize key
        normalized_key = self._normalize_key(key)
        
        # Serialize value
        serialized_value = self._serialize(value, serializer)
        
        # Determine cache levels
        if cache_levels is None:
            cache_levels = self._determine_cache_levels(serialized_value)
        
        success = True
        
        # L1: Set in memory cache
        if CacheLevel.L1_MEMORY in cache_levels:
            self._set_in_memory(normalized_key, value, expire)
            self._record_set(CacheLevel.L1_MEMORY)
        
        # L2: Set in Redis cache
        if CacheLevel.L2_REDIS in cache_levels and self._redis_client:
            success &= await self._set_in_redis(
                normalized_key,
                serialized_value,
                expire
            )
            if success:
                self._record_set(CacheLevel.L2_REDIS)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from all cache levels.
        """
        normalized_key = self._normalize_key(key)
        success = True
        
        # L1: Delete from memory
        if normalized_key in self.memory_cache:
            del self.memory_cache[normalized_key]
            self._record_delete(CacheLevel.L1_MEMORY)
        
        if normalized_key in self.lfu_cache:
            del self.lfu_cache[normalized_key]
        
        # L2: Delete from Redis
        if self._redis_client:
            try:
                await self._redis_client.delete(normalized_key)
                self._record_delete(CacheLevel.L2_REDIS)
            except Exception as e:
                logger.error(f"Failed to delete from Redis: {e}")
                success = False
        
        # Call invalidation callbacks
        for callback in self._invalidation_callbacks:
            try:
                await callback(key)
            except Exception as e:
                logger.error(f"Invalidation callback failed: {e}")
        
        return success
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        """
        count = 0
        
        # L1: Delete from memory cache
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if self._match_pattern(k, pattern)
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]
            count += 1
        
        # L2: Delete from Redis
        if self._redis_client:
            try:
                # Use SCAN to find matching keys
                cursor = 0
                while True:
                    cursor, keys = await self._redis_client.scan(
                        cursor,
                        match=pattern,
                        count=100
                    )
                    
                    if keys:
                        await self._redis_client.delete(*keys)
                        count += len(keys)
                    
                    if cursor == 0:
                        break
                        
            except Exception as e:
                logger.error(f"Failed to delete pattern from Redis: {e}")
        
        return count
    
    async def clear(self) -> bool:
        """
        Clear all cache levels.
        """
        try:
            # L1: Clear memory cache
            self.memory_cache.clear()
            self.lfu_cache.clear()
            
            # L2: Clear Redis
            if self._redis_client:
                await self._redis_client.flushdb()
            
            # Reset statistics
            for level in self.stats:
                self.stats[level] = CacheStats()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def get_many(
        self,
        keys: List[str],
        deserializer: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Get multiple values efficiently.
        """
        results = {}
        missing_keys = []
        
        # Normalize keys
        key_map = {self._normalize_key(k): k for k in keys}
        
        # L1: Check memory cache
        for norm_key, orig_key in key_map.items():
            value = self._get_from_memory(norm_key)
            if value is not None:
                results[orig_key] = value
                self._record_hit(CacheLevel.L1_MEMORY)
            else:
                missing_keys.append(norm_key)
                self._record_miss(CacheLevel.L1_MEMORY)
        
        # L2: Check Redis for missing keys
        if missing_keys and self._redis_client:
            try:
                # Use pipeline for efficiency
                pipe = self._redis_client.pipeline()
                for key in missing_keys:
                    pipe.get(key)
                
                redis_results = await pipe.execute()
                
                for i, value in enumerate(redis_results):
                    if value is not None:
                        norm_key = missing_keys[i]
                        orig_key = next(
                            k for k, v in key_map.items() if v == norm_key
                        )
                        
                        deserialized = self._deserialize(value, deserializer)
                        results[orig_key] = deserialized
                        
                        # Promote to L1
                        self._set_in_memory(norm_key, deserialized)
                        
                        self._record_hit(CacheLevel.L2_REDIS)
                    else:
                        self._record_miss(CacheLevel.L2_REDIS)
                        
            except Exception as e:
                logger.error(f"Failed to get many from Redis: {e}")
        
        return results
    
    async def set_many(
        self,
        items: Dict[str, Any],
        expire: Optional[int] = None,
        serializer: Optional[Callable] = None
    ) -> bool:
        """
        Set multiple values efficiently.
        """
        try:
            # L1: Set in memory cache
            for key, value in items.items():
                norm_key = self._normalize_key(key)
                self._set_in_memory(norm_key, value, expire)
                self._record_set(CacheLevel.L1_MEMORY)
            
            # L2: Set in Redis using pipeline
            if self._redis_client:
                pipe = self._redis_client.pipeline()
                
                for key, value in items.items():
                    norm_key = self._normalize_key(key)
                    serialized = self._serialize(value, serializer)
                    
                    if expire:
                        pipe.setex(norm_key, expire, serialized)
                    else:
                        pipe.set(norm_key, serialized)
                
                await pipe.execute()
                
                for _ in items:
                    self._record_set(CacheLevel.L2_REDIS)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set many: {e}")
            return False
    
    def add_invalidation_callback(self, callback: Callable):
        """
        Add callback for cache invalidation events.
        """
        self._invalidation_callbacks.append(callback)
    
    async def warm_cache(self, keys: List[str], loader: Callable):
        """
        Warm cache with pre-loaded data.
        """
        try:
            # Load data for all keys
            data = await loader(keys)
            
            # Set in cache
            await self.set_many(data)
            
            logger.info(f"Warmed cache with {len(data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to warm cache: {e}")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics.
        """
        return {
            level.value: {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "sets": stats.sets,
                "deletes": stats.deletes,
                "evictions": stats.evictions
            }
            for level, stats in self.stats.items()
        }
    
    # Private methods
    
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key."""
        return f"linkedin:{key}"
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get from memory cache."""
        # Check TTL cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check LFU cache for frequently accessed items
        if key in self.lfu_cache:
            return self.lfu_cache[key]
        
        return None
    
    def _set_in_memory(self, key: str, value: Any, expire: Optional[int] = None):
        """Set in memory cache."""
        self.memory_cache[key] = value
        
        # Also set in LFU cache if accessed frequently
        if hasattr(value, "__sizeof__") and value.__sizeof__() < 1024:
            self.lfu_cache[key] = value
    
    async def _get_from_redis(self, key: str) -> Optional[bytes]:
        """Get from Redis cache."""
        try:
            return await self._redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    async def _set_in_redis(
        self,
        key: str,
        value: bytes,
        expire: Optional[int] = None
    ) -> bool:
        """Set in Redis cache."""
        try:
            if expire:
                await self._redis_client.setex(key, expire, value)
            else:
                await self._redis_client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            return False
    
    def _serialize(self, value: Any, serializer: Optional[Callable] = None) -> bytes:
        """Serialize value for storage."""
        if serializer:
            serialized = serializer(value)
        else:
            # Use orjson for JSON-serializable objects
            try:
                serialized = orjson.dumps(value)
            except:
                # Fall back to pickle for complex objects
                serialized = pickle.dumps(value)
        
        # Compress if above threshold
        if len(serialized) > self.compression_threshold:
            serialized = zlib.compress(serialized)
        
        return serialized
    
    def _deserialize(
        self,
        value: bytes,
        deserializer: Optional[Callable] = None
    ) -> Any:
        """Deserialize value from storage."""
        # Decompress if needed
        try:
            value = zlib.decompress(value)
        except:
            pass  # Not compressed
        
        if deserializer:
            return deserializer(value)
        
        # Try orjson first
        try:
            return orjson.loads(value)
        except:
            # Fall back to pickle
            try:
                return pickle.loads(value)
            except:
                # Return as string if all else fails
                return value.decode('utf-8', errors='ignore')
    
    def _determine_cache_levels(self, value: bytes) -> List[CacheLevel]:
        """Determine which cache levels to use based on value."""
        size = len(value)
        
        # Small values: all levels
        if size < 1024:  # 1KB
            return [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        # Medium values: skip memory for space efficiency
        elif size < 1024 * 100:  # 100KB
            return [CacheLevel.L2_REDIS]
        
        # Large values: Redis only with compression
        else:
            return [CacheLevel.L2_REDIS]
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Match key against pattern with wildcards."""
        return fnmatch.fnmatch(key, pattern)
    
    def _record_hit(self, level: CacheLevel):
        """Record cache hit."""
        if self.enable_stats:
            self.stats[level].hits += 1
    
    def _record_miss(self, level: CacheLevel):
        """Record cache miss."""
        if self.enable_stats:
            self.stats[level].misses += 1
    
    def _record_set(self, level: CacheLevel):
        """Record cache set."""
        if self.enable_stats:
            self.stats[level].sets += 1
    
    def _record_delete(self, level: CacheLevel):
        """Record cache delete."""
        if self.enable_stats:
            self.stats[level].deletes += 1


# Singleton instance
cache_manager = CacheManager()


# Decorator for function caching
def cached(
    expire: int = 300,
    key_prefix: str = "",
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    """
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = await cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set(cache_key, result, expire=expire)
            
            return result
        
        return wrapper
    return decorator


# Export
__all__ = [
    "CacheManager",
    "cache_manager",
    "cached",
    "CacheLevel",
    "CacheStats"
] 