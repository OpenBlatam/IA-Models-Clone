from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import hashlib
import time
from typing import Optional, Any, Dict, List
from functools import wraps
import logging
    import orjson
    import json as orjson
    import redis.asyncio as aioredis
    import diskcache
from cachetools import TTLCache, LRUCache
import structlog
from .config import get_config
                import json
                            import json
                import json
from typing import Any, List, Dict, Optional
"""
High-Performance Cache Manager for Copywriting Service.

Multi-level caching with Redis, memory, and disk storage.
"""


# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    DISK_CACHE_AVAILABLE = True
except ImportError:
    DISK_CACHE_AVAILABLE = False



logger = structlog.get_logger(__name__)

class CacheManager:
    """Ultra-optimized multi-level cache manager."""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.redis_client: Optional[aioredis.Redis] = None
        self.memory_cache: Optional[TTLCache] = None
        self.disk_cache: Optional[diskcache.Cache] = None
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        
        # Initialize caches
        asyncio.create_task(self._initialize_caches())
    
    async def _initialize_caches(self) -> Any:
        """Initialize all cache layers."""
        try:
            # Memory cache
            if self.config.enable_memory_cache:
                self.memory_cache = TTLCache(
                    maxsize=self.config.memory_cache_size,
                    ttl=self.config.cache_ttl_medium
                )
                logger.info("Memory cache initialized", 
                           size=self.config.memory_cache_size)
            
            # Redis cache
            if self.config.enable_redis_cache and REDIS_AVAILABLE:
                self.redis_client = await aioredis.from_url(
                    self.config.redis_url,
                    password=self.config.redis_password,
                    ssl=self.config.redis_ssl,
                    max_connections=self.config.redis_pool_size,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized", 
                           url=self.config.redis_url)
            
            # Disk cache
            if self.config.enable_disk_cache and DISK_CACHE_AVAILABLE:
                self.disk_cache = diskcache.Cache(
                    directory="./cache",
                    size_limit=self.config.disk_cache_size_mb * 1024 * 1024
                )
                logger.info("Disk cache initialized", 
                           size_mb=self.config.disk_cache_size_mb)
                
        except Exception as e:
            logger.error("Failed to initialize caches", error=str(e))
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate optimized cache key."""
        if isinstance(data, str):
            key_data = data
        elif isinstance(data, dict):
            if JSON_AVAILABLE:
                key_data = orjson.dumps(data, sort_keys=True).decode()
            else:
                key_data = json.dumps(data, sort_keys=True)
        else:
            key_data = str(data)
        
        # Fast hash generation
        hash_obj = hashlib.md5(key_data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get value from cache with multi-level fallback."""
        start_time = time.perf_counter()
        
        try:
            # Level 1: Memory cache (fastest)
            if self.memory_cache and key in self.memory_cache:
                self.cache_hits += 1
                value = self.memory_cache[key]
                logger.debug("Cache hit (memory)", key=key, 
                           duration_ms=(time.perf_counter() - start_time) * 1000)
                return value
            
            # Level 2: Redis cache
            if self.redis_client:
                try:
                    value = await self.redis_client.get(key)
                    if value is not None:
                        self.cache_hits += 1
                        # Deserialize
                        if JSON_AVAILABLE:
                            deserialized = orjson.loads(value)
                        else:
                            deserialized = json.loads(value)
                        
                        # Promote to memory cache
                        if self.memory_cache:
                            self.memory_cache[key] = deserialized
                        
                        logger.debug("Cache hit (redis)", key=key,
                                   duration_ms=(time.perf_counter() - start_time) * 1000)
                        return deserialized
                except Exception as e:
                    logger.warning("Redis cache get failed", key=key, error=str(e))
            
            # Level 3: Disk cache (slowest)
            if self.disk_cache:
                try:
                    value = self.disk_cache.get(key)
                    if value is not None:
                        self.cache_hits += 1
                        
                        # Promote to higher levels
                        if self.memory_cache:
                            self.memory_cache[key] = value
                        
                        if self.redis_client:
                            await self._set_redis(key, value, self.config.cache_ttl_medium)
                        
                        logger.debug("Cache hit (disk)", key=key,
                                   duration_ms=(time.perf_counter() - start_time) * 1000)
                        return value
                except Exception as e:
                    logger.warning("Disk cache get failed", key=key, error=str(e))
            
            # Cache miss
            self.cache_misses += 1
            logger.debug("Cache miss", key=key,
                       duration_ms=(time.perf_counter() - start_time) * 1000)
            return default
            
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache levels."""
        if ttl is None:
            ttl = self.config.cache_ttl_medium
        
        success = False
        
        try:
            # Memory cache
            if self.memory_cache:
                self.memory_cache[key] = value
                success = True
            
            # Redis cache
            if self.redis_client:
                success = await self._set_redis(key, value, ttl) or success
            
            # Disk cache
            if self.disk_cache:
                try:
                    self.disk_cache.set(key, value, expire=ttl)
                    success = True
                except Exception as e:
                    logger.warning("Disk cache set failed", key=key, error=str(e))
            
            if success:
                self.cache_sets += 1
                logger.debug("Cache set", key=key, ttl=ttl)
            
            return success
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def _set_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis cache."""
        try:
            # Serialize
            if JSON_AVAILABLE:
                serialized = orjson.dumps(value)
            else:
                serialized = json.dumps(value)
            
            await self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning("Redis cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        success = False
        
        try:
            # Memory cache
            if self.memory_cache and key in self.memory_cache:
                del self.memory_cache[key]
                success = True
            
            # Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.delete(key)
                    success = True
                except Exception as e:
                    logger.warning("Redis delete failed", key=key, error=str(e))
            
            # Disk cache
            if self.disk_cache:
                try:
                    self.disk_cache.delete(key)
                    success = True
                except Exception as e:
                    logger.warning("Disk delete failed", key=key, error=str(e))
            
            return success
            
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all caches."""
        success = False
        
        try:
            # Memory cache
            if self.memory_cache:
                self.memory_cache.clear()
                success = True
            
            # Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                    success = True
                except Exception as e:
                    logger.warning("Redis clear failed", error=str(e))
            
            # Disk cache
            if self.disk_cache:
                try:
                    self.disk_cache.clear()
                    success = True
                except Exception as e:
                    logger.warning("Disk clear failed", error=str(e))
            
            logger.info("All caches cleared")
            return success
            
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_sets": self.cache_sets,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }
        
        # Memory cache stats
        if self.memory_cache:
            stats["memory_cache"] = {
                "size": len(self.memory_cache),
                "maxsize": self.memory_cache.maxsize,
                "currsize": self.memory_cache.currsize
            }
        
        # Disk cache stats
        if self.disk_cache:
            stats["disk_cache"] = {
                "size": len(self.disk_cache),
                "volume": self.disk_cache.volume()
            }
        
        return stats
    
    async def cleanup(self) -> Any:
        """Cleanup cache resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.disk_cache:
                self.disk_cache.close()
                
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error("Cache cleanup error", error=str(e))

# Cache decorators
def cached(prefix: str = "default", ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            cache_manager = get_cache_manager()
            
            # Generate cache key
            key_data = {"args": args, "kwargs": kwargs}
            cache_key = cache_manager._generate_key(f"{prefix}:{func.__name__}", key_data)
            
            # Try to get from cache
            result = await cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

# Export cache utilities
__all__ = [
    "CacheManager",
    "cached",
    "get_cache_manager"
] 