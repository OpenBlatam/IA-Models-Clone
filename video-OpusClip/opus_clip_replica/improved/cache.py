"""
Caching Layer for OpusClip Improved
==================================

Advanced caching system with Redis backend and multiple cache strategies.
"""

import asyncio
import logging
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

from .schemas import get_settings
from .exceptions import CacheError, create_cache_error

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


@dataclass
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy
    ttl: int = 3600  # seconds
    max_size: Optional[int] = None
    compression: bool = False
    serialization: str = "json"  # json, pickle, msgpack


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0


class CacheManager:
    """Advanced cache manager with multiple strategies"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.local_cache = {}  # In-memory cache
        self.stats = CacheStats()
        self._initialize_redis()
        
        # Cache configurations for different data types
        self.cache_configs = {
            "video_analysis": CacheConfig(
                strategy=CacheStrategy.TTL,
                ttl=7200,  # 2 hours
                compression=True,
                serialization="pickle"
            ),
            "clip_generation": CacheConfig(
                strategy=CacheStrategy.TTL,
                ttl=3600,  # 1 hour
                compression=True,
                serialization="pickle"
            ),
            "user_sessions": CacheConfig(
                strategy=CacheStrategy.TTL,
                ttl=1800,  # 30 minutes
                compression=False,
                serialization="json"
            ),
            "api_responses": CacheConfig(
                strategy=CacheStrategy.LRU,
                ttl=300,  # 5 minutes
                compression=False,
                serialization="json"
            ),
            "analytics": CacheConfig(
                strategy=CacheStrategy.TTL,
                ttl=900,  # 15 minutes
                compression=True,
                serialization="pickle"
            ),
            "system_metrics": CacheConfig(
                strategy=CacheStrategy.TTL,
                ttl=60,  # 1 minute
                compression=False,
                serialization="json"
            )
        }
    
    def _initialize_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=False  # Keep binary for pickle
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis for caching: {e}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (dict, list)):
                key_parts.append(f"{k}:{hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()}")
            else:
                key_parts.append(f"{k}:{v}")
        
        return ":".join(key_parts)
    
    def _serialize(self, data: Any, config: CacheConfig) -> bytes:
        """Serialize data for caching"""
        try:
            if config.serialization == "json":
                return json.dumps(data, default=str).encode("utf-8")
            elif config.serialization == "pickle":
                return pickle.dumps(data)
            elif config.serialization == "msgpack":
                import msgpack
                return msgpack.packb(data, default=str)
            else:
                raise ValueError(f"Unsupported serialization: {config.serialization}")
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise create_cache_error("serialization", "cache", e)
    
    def _deserialize(self, data: bytes, config: CacheConfig) -> Any:
        """Deserialize cached data"""
        try:
            if config.serialization == "json":
                return json.loads(data.decode("utf-8"))
            elif config.serialization == "pickle":
                return pickle.loads(data)
            elif config.serialization == "msgpack":
                import msgpack
                return msgpack.unpackb(data, raw=False)
            else:
                raise ValueError(f"Unsupported serialization: {config.serialization}")
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise create_cache_error("deserialization", "cache", e)
    
    async def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """Get value from cache"""
        try:
            config = self.cache_configs.get(cache_type, CacheConfig(strategy=CacheStrategy.TTL))
            
            # Try Redis first
            if self.redis_client:
                try:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        self.stats.hits += 1
                        return self._deserialize(cached_data, config)
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
            
            # Try local cache
            if key in self.local_cache:
                cached_item = self.local_cache[key]
                if cached_item["expires"] > datetime.utcnow():
                    self.stats.hits += 1
                    return cached_item["data"]
                else:
                    # Expired, remove from local cache
                    del self.local_cache[key]
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self.stats.misses += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "default",
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            config = self.cache_configs.get(cache_type, CacheConfig(strategy=CacheStrategy.TTL))
            
            # Use provided TTL or config TTL
            cache_ttl = ttl or config.ttl
            
            # Serialize data
            serialized_data = self._serialize(value, config)
            
            # Store in Redis
            if self.redis_client:
                try:
                    await self.redis_client.setex(key, cache_ttl, serialized_data)
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
            
            # Store in local cache
            self.local_cache[key] = {
                "data": value,
                "expires": datetime.utcnow() + timedelta(seconds=cache_ttl),
                "created": datetime.utcnow()
            }
            
            # Clean up expired local cache entries
            await self._cleanup_local_cache()
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Delete from Redis
            if self.redis_client:
                try:
                    await self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete failed: {e}")
            
            # Delete from local cache
            if key in self.local_cache:
                del self.local_cache[key]
            
            self.stats.deletes += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            # Check Redis
            if self.redis_client:
                try:
                    return await self.redis_client.exists(key) > 0
                except Exception as e:
                    logger.warning(f"Redis exists check failed: {e}")
            
            # Check local cache
            if key in self.local_cache:
                cached_item = self.local_cache[key]
                if cached_item["expires"] > datetime.utcnow():
                    return True
                else:
                    # Expired, remove from local cache
                    del self.local_cache[key]
            
            return False
            
        except Exception as e:
            logger.error(f"Cache exists check failed: {e}")
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        try:
            deleted_count = 0
            
            # Clear Redis
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        deleted_count += await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
            
            # Clear local cache
            if pattern == "*":
                self.local_cache.clear()
            else:
                # Simple pattern matching for local cache
                keys_to_delete = [k for k in self.local_cache.keys() if pattern.replace("*", "") in k]
                for key in keys_to_delete:
                    del self.local_cache[key]
                deleted_count += len(keys_to_delete)
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        cache_type: str = "default",
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or set it using factory function"""
        try:
            # Try to get from cache
            cached_value = await self.get(key, cache_type)
            if cached_value is not None:
                return cached_value
            
            # Generate value using factory
            value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            
            # Cache the value
            await self.set(key, value, cache_type, ttl)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get_or_set failed: {e}")
            # Return factory result even if caching fails
            return await factory() if asyncio.iscoroutinefunction(factory) else factory()
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        return await self.clear(pattern)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "sets": self.stats.sets,
                "deletes": self.stats.deletes,
                "evictions": self.stats.evictions,
                "hit_rate": round(hit_rate, 2),
                "local_cache_size": len(self.local_cache),
                "total_requests": total_requests
            }
            
            # Get Redis stats if available
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info("memory")
                    stats["redis_memory_usage"] = redis_info.get("used_memory", 0)
                    stats["redis_memory_peak"] = redis_info.get("used_memory_peak", 0)
                    
                    # Count Redis keys
                    redis_keys = await self.redis_client.dbsize()
                    stats["redis_keys"] = redis_keys
                except Exception as e:
                    logger.warning(f"Failed to get Redis stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    async def _cleanup_local_cache(self):
        """Clean up expired entries from local cache"""
        try:
            now = datetime.utcnow()
            expired_keys = [
                key for key, item in self.local_cache.items()
                if item["expires"] <= now
            ]
            
            for key in expired_keys:
                del self.local_cache[key]
                self.stats.evictions += 1
            
            # Limit local cache size
            max_local_size = 1000
            if len(self.local_cache) > max_local_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.local_cache.items(),
                    key=lambda x: x[1]["created"]
                )
                
                items_to_remove = len(self.local_cache) - max_local_size
                for key, _ in sorted_items[:items_to_remove]:
                    del self.local_cache[key]
                    self.stats.evictions += 1
            
        except Exception as e:
            logger.error(f"Local cache cleanup failed: {e}")
    
    async def warm_cache(self, warmup_data: Dict[str, Any]):
        """Warm up cache with predefined data"""
        try:
            for key, (value, cache_type, ttl) in warmup_data.items():
                await self.set(key, value, cache_type, ttl)
            
            logger.info(f"Cache warmed up with {len(warmup_data)} entries")
            
        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        try:
            health = {
                "status": "healthy",
                "redis_connected": False,
                "local_cache_working": False,
                "errors": []
            }
            
            # Test Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health["redis_connected"] = True
                except Exception as e:
                    health["errors"].append(f"Redis connection failed: {e}")
            
            # Test local cache
            try:
                test_key = "health_check_test"
                test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}
                
                await self.set(test_key, test_value, "default", 60)
                retrieved_value = await self.get(test_key)
                
                if retrieved_value and retrieved_value.get("test"):
                    health["local_cache_working"] = True
                    await self.delete(test_key)
                else:
                    health["errors"].append("Local cache read/write test failed")
                    
            except Exception as e:
                health["errors"].append(f"Local cache test failed: {e}")
            
            # Overall health status
            if health["errors"]:
                health["status"] = "unhealthy"
            elif not health["redis_connected"] and not health["local_cache_working"]:
                health["status"] = "unhealthy"
            
            return health
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global cache manager
cache_manager = CacheManager()


# Cache decorators
def cached(
    cache_type: str = "default",
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_func: Optional[Callable] = None
):
    """Cache decorator for functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Use function name and arguments
                key_parts = [key_prefix, func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, cache_type, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(pattern: str):
    """Cache invalidation decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Invalidate cache
            await cache_manager.invalidate_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator


# Cache utilities
class CacheUtils:
    """Cache utility functions"""
    
    @staticmethod
    def generate_key_from_request(request_data: Dict[str, Any]) -> str:
        """Generate cache key from request data"""
        key_parts = []
        
        for key, value in sorted(request_data.items()):
            if isinstance(value, (dict, list)):
                key_parts.append(f"{key}:{hashlib.md5(json.dumps(value, sort_keys=True).encode()).hexdigest()}")
            else:
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def should_cache_response(status_code: int, content_type: str) -> bool:
        """Determine if response should be cached"""
        # Cache successful responses
        if status_code not in [200, 201, 202]:
            return False
        
        # Cache JSON and text responses
        cacheable_types = ["application/json", "text/plain", "text/html"]
        return any(ct in content_type for ct in cacheable_types)
    
    @staticmethod
    def get_cache_ttl_for_endpoint(endpoint: str) -> int:
        """Get appropriate TTL for endpoint"""
        ttl_map = {
            "/api/v2/opus-clip/health": 60,
            "/api/v2/opus-clip/stats": 300,
            "/api/v2/opus-clip/analytics": 900,
            "/api/v2/opus-clip/projects": 600,
        }
        
        return ttl_map.get(endpoint, 300)  # Default 5 minutes





























