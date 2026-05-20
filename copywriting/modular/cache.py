from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Optional, Any, Dict
from functools import lru_cache
    import orjson
    import json as orjson
    import redis.asyncio as aioredis
    import hiredis
        import redis.asyncio as aioredis
    import xxhash
    import lz4.frame
import structlog
from .config import get_config
            import hashlib
            import json
            import json
            import hashlib
from typing import Any, List, Dict, Optional
import logging
"""
Modular Cache Manager with High-Performance Libraries.

Multi-level caching with optimized libraries:
- orjson for ultra-fast JSON serialization
- redis with hiredis for ultra-fast Redis protocol
- lz4 for fast compression
- xxhash for fast hashing
"""


# High-performance imports with fallbacks
try:
    JSON_LIB = "orjson"
except ImportError:
    JSON_LIB = "json"

try:
    REDIS_AVAILABLE = True
    HIREDIS_AVAILABLE = True
except ImportError:
    try:
        REDIS_AVAILABLE = True
        HIREDIS_AVAILABLE = False
    except ImportError:
        REDIS_AVAILABLE = False
        HIREDIS_AVAILABLE = False

try:
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

try:
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


logger = structlog.get_logger(__name__)

class OptimizedCacheManager:
    """High-performance cache manager with multiple optimization libraries."""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.redis_client: Optional[aioredis.Redis] = None
        self.memory_cache: Dict[str, Any] = {}
        
        # Performance stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        
        # Optimization flags
        self.optimizations = {
            "json": JSON_LIB,
            "redis": REDIS_AVAILABLE,
            "hiredis": HIREDIS_AVAILABLE,
            "xxhash": XXHASH_AVAILABLE,
            "lz4": LZ4_AVAILABLE
        }
        
        logger.info("OptimizedCacheManager initialized", optimizations=self.optimizations)
    
    async def initialize(self) -> Any:
        """Initialize Redis connection with optimizations."""
        if not REDIS_AVAILABLE or not self.config.enable_cache:
            logger.info("Redis caching disabled")
            return
        
        try:
            connection_kwargs = {
                "encoding": "utf-8",
                "decode_responses": False,  # We handle bytes for compression
                "max_connections": 20
            }
            
            # Use hiredis for ultra-fast protocol parsing
            if HIREDIS_AVAILABLE:
                connection_kwargs["connection_class"] = aioredis.Connection
                logger.info("Using hiredis for ultra-fast Redis protocol")
            
            self.redis_client = await aioredis.from_url(
                self.config.redis_url,
                **connection_kwargs
            )
            
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully", 
                       url=self.config.redis_url,
                       hiredis=HIREDIS_AVAILABLE)
            
        except Exception as e:
            logger.warning("Redis initialization failed", error=str(e))
            self.redis_client = None
    
    def _fast_hash(self, data: str) -> str:
        """Ultra-fast hashing with xxhash."""
        if XXHASH_AVAILABLE:
            return xxhash.xxh64(data.encode()).hexdigest()
        else:
            return hashlib.md5(data.encode()).hexdigest()
    
    def _fast_compress(self, data: bytes) -> bytes:
        """Ultra-fast compression with lz4."""
        if LZ4_AVAILABLE and len(data) > 100:  # Only compress larger data
            try:
                return lz4.frame.compress(data)
            except Exception:
                return data
        return data
    
    def _fast_decompress(self, data: bytes) -> bytes:
        """Ultra-fast decompression with lz4."""
        if LZ4_AVAILABLE:
            try:
                return lz4.frame.decompress(data)
            except Exception:
                return data
        return data
    
    def _fast_serialize(self, obj: Any) -> bytes:
        """Ultra-fast serialization with orjson."""
        if JSON_LIB == "orjson":
            return orjson.dumps(obj)
        else:
            return json.dumps(obj).encode()
    
    def _fast_deserialize(self, data: bytes) -> Any:
        """Ultra-fast deserialization with orjson."""
        if JSON_LIB == "orjson":
            return orjson.loads(data)
        else:
            return json.loads(data.decode())
    
    async def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get value from cache with ultra-fast operations."""
        start_time = time.perf_counter()
        
        try:
            # Level 1: Memory cache (fastest)
            if key in self.memory_cache:
                self.cache_hits += 1
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.debug("Memory cache hit", key=key, duration_ms=duration_ms)
                return self.memory_cache[key]
            
            # Level 2: Redis cache with optimizations
            if self.redis_client:
                try:
                    cache_key = self._fast_hash(key)
                    compressed_data = await self.redis_client.get(cache_key)
                    
                    if compressed_data:
                        # Ultra-fast decompression and deserialization
                        decompressed_data = self._fast_decompress(compressed_data)
                        result = self._fast_deserialize(decompressed_data)
                        
                        # Promote to memory cache
                        self.memory_cache[key] = result
                        self.cache_hits += 1
                        
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        logger.debug("Redis cache hit", key=key, duration_ms=duration_ms)
                        return result
                        
                except Exception as e:
                    logger.warning("Redis cache get failed", key=key, error=str(e))
            
            # Cache miss
            self.cache_misses += 1
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("Cache miss", key=key, duration_ms=duration_ms)
            return default
            
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with ultra-fast operations."""
        if ttl is None:
            ttl = self.config.cache_ttl
        
        start_time = time.perf_counter()
        success = False
        
        try:
            # Memory cache
            self.memory_cache[key] = value
            success = True
            
            # Redis cache with optimizations
            if self.redis_client:
                try:
                    # Ultra-fast serialization and compression
                    serialized_data = self._fast_serialize(value)
                    compressed_data = self._fast_compress(serialized_data)
                    
                    cache_key = self._fast_hash(key)
                    await self.redis_client.setex(cache_key, ttl, compressed_data)
                    
                    success = True
                    
                except Exception as e:
                    logger.warning("Redis cache set failed", key=key, error=str(e))
            
            if success:
                self.cache_sets += 1
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.debug("Cache set", key=key, ttl=ttl, duration_ms=duration_ms)
            
            return success
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        success = False
        
        try:
            # Memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                success = True
            
            # Redis cache
            if self.redis_client:
                try:
                    cache_key = self._fast_hash(key)
                    await self.redis_client.delete(cache_key)
                    success = True
                except Exception as e:
                    logger.warning("Redis delete failed", key=key, error=str(e))
            
            return success
            
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all caches."""
        try:
            # Memory cache
            self.memory_cache.clear()
            
            # Redis cache (flush database)
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.warning("Redis clear failed", error=str(e))
            
            logger.info("All caches cleared")
            return True
            
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_sets": self.cache_sets,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
            "optimizations": self.optimizations
        }
    
    def get_optimization_info(self) -> Dict[str, str]:
        """Get optimization library information."""
        info = {}
        
        if JSON_LIB == "orjson":
            info["json"] = "orjson (5x faster)"
        else:
            info["json"] = "standard json"
        
        if REDIS_AVAILABLE:
            if HIREDIS_AVAILABLE:
                info["redis"] = "redis + hiredis (ultra-fast protocol)"
            else:
                info["redis"] = "redis (standard)"
        else:
            info["redis"] = "not available"
        
        if XXHASH_AVAILABLE:
            info["hashing"] = "xxhash (4x faster)"
        else:
            info["hashing"] = "md5 (standard)"
        
        if LZ4_AVAILABLE:
            info["compression"] = "lz4 (3x faster)"
        else:
            info["compression"] = "none"
        
        return info
    
    async def cleanup(self) -> Any:
        """Cleanup cache resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.memory_cache.clear()
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error("Cache cleanup error", error=str(e))

# Global cache manager instance
_cache_manager: Optional[OptimizedCacheManager] = None

@lru_cache(maxsize=1)
def get_cache_manager() -> OptimizedCacheManager:
    """Get cached cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = OptimizedCacheManager()
    return _cache_manager

async def initialize_cache():
    """Initialize cache manager."""
    cache_manager = get_cache_manager()
    await cache_manager.initialize()

# Cache decorator for functions
def cached(key_prefix: str = "func", ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            cache_manager = get_cache_manager()
            
            # Generate cache key
            key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache first
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

# Export cache utilities
__all__ = [
    "OptimizedCacheManager",
    "get_cache_manager", 
    "initialize_cache",
    "cached"
] 