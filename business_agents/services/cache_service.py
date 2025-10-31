"""
Cache Service
=============

Advanced caching service with Redis support, memory caching, and intelligent invalidation.
"""

import asyncio
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
from functools import wraps
import redis
import aioredis
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration."""
    redis_url: Optional[str] = None
    memory_cache_size: int = 1000
    default_ttl: int = 3600
    compression_enabled: bool = True
    serialization_method: str = "pickle"  # pickle, json
    key_prefix: str = "business_agents"
    enable_memory_cache: bool = True
    enable_redis_cache: bool = True

class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
    def _evict_lru(self):
        """Evict least recently used items."""
        if len(self.cache) >= self.max_size:
            # Find least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            self._evict_lru()
            self.cache[key] = value
            self.access_times[key] = time.time()
            return True
            
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False
            
    def clear(self) -> int:
        """Clear all cache entries."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            return count
            
    def size(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)
            
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            return list(self.cache.keys())

class CacheService:
    """
    Advanced caching service with multi-level caching support.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config.memory_cache_size) if config.enable_memory_cache else None
        self.redis_client = None
        self.redis_pool = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize cache service."""
        if self._initialized:
            return
            
        # Initialize Redis if configured
        if self.config.enable_redis_cache and self.config.redis_url:
            try:
                self.redis_pool = aioredis.ConnectionPool.from_url(
                    self.config.redis_url,
                    max_connections=20,
                    retry_on_timeout=True
                )
                self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {str(e)}")
                self.redis_client = None
                self.redis_pool = None
                
        self._initialized = True
        
    async def close(self):
        """Close cache service."""
        if self.redis_pool:
            await self.redis_pool.disconnect()
            
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        if self.config.serialization_method == "json":
            return json.dumps(data, default=str).encode('utf-8')
        else:
            return pickle.dumps(data)
            
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        if self.config.serialization_method == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)
            
    def _make_key(self, key: str) -> str:
        """Make full cache key with prefix."""
        return f"{self.config.key_prefix}:{key}"
        
    def _hash_key(self, key: str) -> str:
        """Hash key for consistent length."""
        return hashlib.md5(key.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(hashed_key)
            if value is not None:
                return value
                
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(hashed_key)
                if cached_data:
                    value = self._deserialize(cached_data)
                    # Store in memory cache for faster access
                    if self.memory_cache:
                        self.memory_cache.set(hashed_key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {str(e)}")
                
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        ttl = ttl or self.config.default_ttl
        
        success = True
        
        # Set in memory cache
        if self.memory_cache:
            self.memory_cache.set(hashed_key, value)
            
        # Set in Redis cache
        if self.redis_client:
            try:
                serialized_data = self._serialize(value)
                await self.redis_client.setex(hashed_key, ttl, serialized_data)
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")
                success = False
                
        return success
        
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        
        success = True
        
        # Delete from memory cache
        if self.memory_cache:
            self.memory_cache.delete(hashed_key)
            
        # Delete from Redis cache
        if self.redis_client:
            try:
                await self.redis_client.delete(hashed_key)
            except Exception as e:
                logger.error(f"Redis delete error: {str(e)}")
                success = False
                
        return success
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        
        # Check memory cache
        if self.memory_cache and self.memory_cache.get(hashed_key) is not None:
            return True
            
        # Check Redis cache
        if self.redis_client:
            try:
                return await self.redis_client.exists(hashed_key) > 0
            except Exception as e:
                logger.error(f"Redis exists error: {str(e)}")
                
        return False
        
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        
        if self.redis_client:
            try:
                return await self.redis_client.expire(hashed_key, ttl)
            except Exception as e:
                logger.error(f"Redis expire error: {str(e)}")
                
        return False
        
    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        if not self._initialized:
            await self.initialize()
            
        full_key = self._make_key(key)
        hashed_key = self._hash_key(full_key)
        
        if self.redis_client:
            try:
                return await self.redis_client.ttl(hashed_key)
            except Exception as e:
                logger.error(f"Redis ttl error: {str(e)}")
                
        return -1
        
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self._initialized:
            await self.initialize()
            
        full_pattern = self._make_key(pattern)
        deleted_count = 0
        
        # Clear from memory cache
        if self.memory_cache:
            keys_to_delete = [key for key in self.memory_cache.keys() if pattern in key]
            for key in keys_to_delete:
                self.memory_cache.delete(key)
                deleted_count += 1
                
        # Clear from Redis cache
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(full_pattern)
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear pattern error: {str(e)}")
                
        return deleted_count
        
    async def clear_all(self) -> int:
        """Clear all cache entries."""
        if not self._initialized:
            await self.initialize()
            
        deleted_count = 0
        
        # Clear memory cache
        if self.memory_cache:
            deleted_count += self.memory_cache.clear()
            
        # Clear Redis cache
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"{self.config.key_prefix}:*")
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear all error: {str(e)}")
                
        return deleted_count
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized:
            await self.initialize()
            
        stats = {
            "memory_cache": {
                "enabled": self.memory_cache is not None,
                "size": self.memory_cache.size() if self.memory_cache else 0,
                "max_size": self.config.memory_cache_size
            },
            "redis_cache": {
                "enabled": self.redis_client is not None,
                "connected": False
            }
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["redis_cache"].update({
                    "connected": True,
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100
                })
            except Exception as e:
                stats["redis_cache"]["error"] = str(e)
                
        return stats
        
    def cached(self, key_prefix: str, ttl: Optional[int] = None, use_args: bool = True):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if use_args:
                    key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                else:
                    key_data = f"{key_prefix}:{func.__name__}"
                    
                cache_key = self._hash_key(key_data)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if use_args:
                    key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                else:
                    key_data = f"{key_prefix}:{func.__name__}"
                    
                cache_key = self._hash_key(key_data)
                
                # Try to get from cache (sync)
                loop = asyncio.get_event_loop()
                cached_result = loop.run_until_complete(self.get(cache_key))
                if cached_result is not None:
                    return cached_result
                    
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result (sync)
                loop.run_until_complete(self.set(cache_key, result, ttl))
                
                return result
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
            
        return decorator
        
    async def invalidate_function_cache(self, function_name: str) -> int:
        """Invalidate cache for a specific function."""
        pattern = f"{self.config.key_prefix}:*:{function_name}:*"
        return await self.clear_pattern(pattern)
        
    async def warm_cache(self, warmup_functions: List[Callable]) -> Dict[str, Any]:
        """Warm up cache with predefined functions."""
        results = {}
        
        for func in warmup_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                results[func.__name__] = {"status": "success", "result": result}
            except Exception as e:
                results[func.__name__] = {"status": "error", "error": str(e)}
                
        return results

# Global cache service instance
cache_service = None

def get_cache_service() -> CacheService:
    """Get global cache service instance."""
    global cache_service
    if cache_service is None:
        config = CacheConfig(
            redis_url="redis://localhost:6379",
            memory_cache_size=1000,
            default_ttl=3600,
            compression_enabled=True,
            serialization_method="pickle",
            key_prefix="business_agents",
            enable_memory_cache=True,
            enable_redis_cache=True
        )
        cache_service = CacheService(config)
    return cache_service




























