from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import gzip
import base64
import redis.asyncio as redis
import aioredis
from cachetools import TTLCache, LRUCache
import orjson
import msgpack
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
Advanced Cache Manager
======================

Multi-layer caching system using Redis with intelligent cache management,
compression, and advanced features for LinkedIn posts system.
"""




logger = get_logger(__name__)


class AdvancedCacheManager:
    """
    Advanced cache manager with multiple layers and intelligent features.
    
    Features:
    - Multi-layer caching (L1: Memory, L2: Redis)
    - Compression for large objects
    - Intelligent cache invalidation
    - Cache warming and prefetching
    - Cache analytics and monitoring
    - Distributed locking
    - Cache versioning
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        memory_cache_size: int = 1000,
        memory_cache_ttl: int = 300,
        compression_threshold: int = 1024,
        enable_compression: bool = True,
        enable_analytics: bool = True,
    ):
        """Initialize the advanced cache manager."""
        self.redis_url = redis_url
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        self.enable_analytics = enable_analytics
        
        # Initialize Redis connection
        self.redis_client = None
        self.redis_pool = None
        
        # Memory cache (L1)
        self.memory_cache = TTLCache(
            maxsize=memory_cache_size,
            ttl=memory_cache_ttl
        )
        
        # Cache analytics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "compressions": 0,
            "errors": 0,
        }
        
        # Cache version
        self.cache_version = "1.0.0"
        
        # Initialize connection
        asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=False
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, key: str, namespace: str = "linkedin_posts") -> str:
        """Generate a cache key with namespace and version."""
        return f"{namespace}:{self.cache_version}:{key}"
    
    def _compress_data(self, data: Any) -> Dict[str, Any]:
        """Compress data if it exceeds threshold."""
        if not self.enable_compression:
            return {"data": data, "compressed": False}
        
        try:
            # Serialize data
            serialized = orjson.dumps(data)
            
            if len(serialized) > self.compression_threshold:
                # Compress data
                compressed = gzip.compress(serialized)
                encoded = base64.b64encode(compressed).decode('utf-8')
                
                self.cache_stats["compressions"] += 1
                return {
                    "data": encoded,
                    "compressed": True,
                    "original_size": len(serialized),
                    "compressed_size": len(compressed)
                }
            else:
                return {"data": serialized, "compressed": False}
                
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return {"data": data, "compressed": False}
    
    def _decompress_data(self, cache_data: Dict[str, Any]) -> Any:
        """Decompress data if it was compressed."""
        try:
            if cache_data.get("compressed", False):
                # Decompress data
                encoded = cache_data["data"]
                compressed = base64.b64decode(encoded.encode('utf-8'))
                serialized = gzip.decompress(compressed)
                return orjson.loads(serialized)
            else:
                # Handle both serialized and raw data
                if isinstance(cache_data["data"], bytes):
                    return orjson.loads(cache_data["data"])
                else:
                    return cache_data["data"]
                    
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return cache_data.get("data")
    
    async def get(self, key: str, namespace: str = "linkedin_posts") -> Optional[Any]:
        """Get value from cache (L1 then L2)."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Try L1 cache (memory)
            if cache_key in self.memory_cache:
                self.cache_stats["hits"] += 1
                logger.debug(f"L1 cache hit for key: {key}")
                return self.memory_cache[cache_key]
            
            # Try L2 cache (Redis)
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    # Deserialize and decompress
                    cache_data = orjson.loads(cached_data)
                    data = self._decompress_data(cache_data)
                    
                    # Store in L1 cache
                    self.memory_cache[cache_key] = data
                    
                    self.cache_stats["hits"] += 1
                    logger.debug(f"L2 cache hit for key: {key}")
                    return data
            
            self.cache_stats["misses"] += 1
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        namespace: str = "linkedin_posts",
        compress: Optional[bool] = None
    ) -> bool:
        """Set value in cache (both L1 and L2)."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Compress data if needed
            should_compress = compress if compress is not None else self.enable_compression
            cache_data = self._compress_data(value) if should_compress else {"data": value, "compressed": False}
            
            # Add metadata
            cache_data.update({
                "created_at": datetime.utcnow().isoformat(),
                "ttl": ttl,
                "version": self.cache_version
            })
            
            # Store in L1 cache (memory)
            self.memory_cache[cache_key] = value
            
            # Store in L2 cache (Redis)
            if self.redis_client:
                serialized = orjson.dumps(cache_data)
                await self.redis_client.setex(cache_key, ttl, serialized)
            
            self.cache_stats["sets"] += 1
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str, namespace: str = "linkedin_posts") -> bool:
        """Delete value from cache (both L1 and L2)."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Remove from L1 cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Remove from L2 cache
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            self.cache_stats["deletes"] += 1
            logger.debug(f"Cache delete for key: {key}")
            return True
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = "linkedin_posts") -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Check L1 cache
            if cache_key in self.memory_cache:
                return True
            
            # Check L2 cache
            if self.redis_client:
                return await self.redis_client.exists(cache_key) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False
    
    async def expire(self, key: str, ttl: int, namespace: str = "linkedin_posts") -> bool:
        """Set expiration for cache key."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Update L1 cache TTL
            if cache_key in self.memory_cache:
                # Re-insert with new TTL
                value = self.memory_cache[cache_key]
                del self.memory_cache[cache_key]
                self.memory_cache[cache_key] = value
            
            # Update L2 cache TTL
            if self.redis_client:
                return await self.redis_client.expire(cache_key, ttl)
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting cache expiration: {e}")
            return False
    
    async def clear_namespace(self, namespace: str = "linkedin_posts") -> bool:
        """Clear all keys in a namespace."""
        try:
            pattern = f"{namespace}:{self.cache_version}:*"
            
            # Clear L1 cache
            keys_to_remove = [key for key in self.memory_cache.keys() if key.startswith(f"{namespace}:{self.cache_version}:")]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # Clear L2 cache
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info(f"Cleared namespace: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing namespace: {e}")
            return False
    
    async def get_many(self, keys: List[str], namespace: str = "linkedin_posts") -> Dict[str, Any]:
        """Get multiple values from cache."""
        results = {}
        
        try:
            for key in keys:
                value = await self.get(key, namespace)
                if value is not None:
                    results[key] = value
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multiple values from cache: {e}")
            return results
    
    async def set_many(
        self,
        data: Dict[str, Any],
        ttl: int = 3600,
        namespace: str = "linkedin_posts"
    ) -> bool:
        """Set multiple values in cache."""
        try:
            tasks = [
                self.set(key, value, ttl, namespace)
                for key, value in data.items()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            
            logger.info(f"Set {success_count}/{len(data)} values in cache")
            return success_count == len(data)
            
        except Exception as e:
            logger.error(f"Error setting multiple values in cache: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, namespace: str = "linkedin_posts") -> Optional[int]:
        """Increment a numeric value in cache."""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            if self.redis_client:
                return await self.redis_client.incrby(cache_key, amount)
            return None
            
        except Exception as e:
            logger.error(f"Error incrementing cache value: {e}")
            return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and analytics."""
        try:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                **self.cache_stats,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
                "memory_cache_size": len(self.memory_cache),
                "memory_cache_maxsize": self.memory_cache.maxsize,
                "redis_connected": self.redis_client is not None,
                "cache_version": self.cache_version,
                "compression_enabled": self.enable_compression,
                "compression_threshold": self.compression_threshold,
            }
            
            # Get Redis info if available
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    stats["redis_info"] = {
                        "used_memory": redis_info.get("used_memory", 0),
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "total_commands_processed": redis_info.get("total_commands_processed", 0),
                    }
                except Exception as e:
                    logger.error(f"Error getting Redis info: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.cache_stats
    
    async def warm_cache(self, warmup_data: Dict[str, Any], namespace: str = "linkedin_posts") -> bool:
        """Warm up cache with predefined data."""
        try:
            logger.info(f"Warming cache with {len(warmup_data)} items")
            return await self.set_many(warmup_data, namespace=namespace)
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return False
    
    async async def prefetch(self, keys: List[str], namespace: str = "linkedin_posts") -> bool:
        """Prefetch keys into L1 cache."""
        try:
            logger.info(f"Prefetching {len(keys)} keys into L1 cache")
            
            for key in keys:
                await self.get(key, namespace)  # This will populate L1 cache
            
            return True
            
        except Exception as e:
            logger.error(f"Error prefetching cache: {e}")
            return False
    
    async def get_cache_keys(self, pattern: str = "*", namespace: str = "linkedin_posts") -> List[str]:
        """Get cache keys matching pattern."""
        try:
            if self.redis_client:
                full_pattern = f"{namespace}:{self.cache_version}:{pattern}"
                keys = await self.redis_client.keys(full_pattern)
                return [key.decode('utf-8') for key in keys]
            return []
            
        except Exception as e:
            logger.error(f"Error getting cache keys: {e}")
            return []
    
    async def get_cache_size(self, namespace: str = "linkedin_posts") -> Dict[str, int]:
        """Get cache size information."""
        try:
            pattern = f"{namespace}:{self.cache_version}:*"
            
            # L1 cache size
            l1_size = len([k for k in self.memory_cache.keys() if k.startswith(f"{namespace}:{self.cache_version}:")])
            
            # L2 cache size
            l2_size = 0
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                l2_size = len(keys)
            
            return {
                "l1_size": l1_size,
                "l2_size": l2_size,
                "total_size": l1_size + l2_size
            }
            
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return {"l1_size": 0, "l2_size": 0, "total_size": 0}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from L1 cache."""
        try:
            # TTLCache handles expiration automatically
            # This is just for logging purposes
            initial_size = len(self.memory_cache)
            # Force cleanup by accessing cache
            _ = list(self.memory_cache.keys())
            final_size = len(self.memory_cache)
            
            cleaned = initial_size - final_size
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired entries from L1 cache")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0
    
    async def close(self) -> Any:
        """Close cache manager and connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            logger.info("Cache manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")


def cache_result(
    ttl: int = 3600,
    namespace: str = "linkedin_posts",
    key_func: Optional[Callable] = None,
    cache_manager: Optional[AdvancedCacheManager] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Cache TTL in seconds
        namespace: Cache namespace
        key_func: Function to generate cache key from function arguments
        cache_manager: Cache manager instance
    """
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Use provided cache manager or create default
            cm = cache_manager or AdvancedCacheManager()
            
            # Try to get from cache
            cached_result = await cm.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cm.set(cache_key, result, ttl, namespace)
            
            return result
        
        return wrapper
    return decorator


class CacheAwareLinkedInPostGenerator:
    """
    Cache-aware LinkedIn post generator with intelligent caching.
    """
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
    
    @cache_result(ttl=1800, namespace="linkedin_posts")
    async def generate_post_with_cache(
        self,
        topic: str,
        key_points: List[str],
        target_audience: str,
        industry: str,
        tone: str,
        post_type: str,
        keywords: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate LinkedIn post with intelligent caching.
        Cache key is based on input parameters for consistent results.
        """
        # This would call the actual post generation logic
        # For now, return mock data
        return {
            "title": f"Generated post about {topic}",
            "content": f"This is a {tone} post about {topic} for {target_audience} in {industry}",
            "hashtags": ["#linkedin", "#content", "#generation"],
            "estimated_engagement": 75.5,
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    async def get_cached_post(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached post by key."""
        return await self.cache_manager.get(cache_key, "linkedin_posts")
    
    async def cache_post_generation(
        self,
        cache_key: str,
        post_data: Dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """Cache post generation result."""
        return await self.cache_manager.set(cache_key, post_data, ttl, "linkedin_posts") 