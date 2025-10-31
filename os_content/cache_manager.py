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
import logging
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, List, Union
from functools import wraps
from pathlib import Path
import json
import orjson
import ujson
from cachetools import TTLCache, LRUCache
from diskcache import Cache as DiskCache
import aioredis
import zstandard as zstd
import lz4.frame
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Cache Manager for OS Content UGC Video Generator
Implements multiple caching strategies for optimal performance
"""


# Performance libraries

logger = logging.getLogger("os_content.cache")

class MultiLevelCache:
    """Multi-level cache with L1 (memory), L2 (disk), L3 (Redis)"""
    
    def __init__(self, 
                 memory_size: int = 1000,
                 memory_ttl: int = 300,
                 disk_size: int = 10000,
                 disk_ttl: int = 3600,
                 redis_url: Optional[str] = None):
        
        
    """__init__ function."""
# L1: Memory cache (fastest)
        self.l1_cache = TTLCache(maxsize=memory_size, ttl=memory_ttl)
        
        # L2: Disk cache (persistent)
        self.l2_cache = DiskCache(directory="./cache/disk", size_limit=disk_size * 1024 * 1024)
        
        # L3: Redis cache (distributed)
        self.redis_client = None
        if redis_url:
            self.redis_client = aioredis.from_url(redis_url)
        
        # Compression settings
        self.compression_threshold = 1024  # bytes
        self.use_compression = True
        
        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "sets": 0
        }
    
    def _generate_key(self, key: Union[str, bytes]) -> str:
        """Generate a consistent cache key"""
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zstandard"""
        if len(data) > self.compression_threshold and self.use_compression:
            return zstd.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using zstandard"""
        try:
            if data.startswith(b'\x28\xb5\x2f\xfd'):  # zstd magic number
                return zstd.decompress(data)
        except Exception:
            pass
        return data
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data using orjson for better performance"""
        try:
            return orjson.dumps(data)
        except Exception:
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data using orjson"""
        try:
            return orjson.loads(data)
        except Exception:
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        cache_key = self._generate_key(key)
        
        # L1: Memory cache
        if cache_key in self.l1_cache:
            self.stats["l1_hits"] += 1
            return self.l1_cache[cache_key]
        
        # L2: Disk cache
        try:
            compressed_data = self.l2_cache.get(cache_key)
            if compressed_data is not None:
                data = self._decompress_data(compressed_data)
                value = self._deserialize_data(data)
                # Promote to L1
                self.l1_cache[cache_key] = value
                self.stats["l2_hits"] += 1
                return value
        except Exception as e:
            logger.warning(f"L2 cache error: {e}")
        
        # L3: Redis cache
        if self.redis_client:
            try:
                compressed_data = await self.redis_client.get(cache_key)
                if compressed_data:
                    data = self._decompress_data(compressed_data)
                    value = self._deserialize_data(data)
                    # Promote to L1 and L2
                    self.l1_cache[cache_key] = value
                    self.l2_cache.set(cache_key, compressed_data)
                    self.stats["l3_hits"] += 1
                    return value
            except Exception as e:
                logger.warning(f"L3 cache error: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multi-level cache"""
        cache_key = self._generate_key(key)
        
        try:
            # Serialize and compress data
            serialized_data = self._serialize_data(value)
            compressed_data = self._compress_data(serialized_data)
            
            # L1: Memory cache
            self.l1_cache[cache_key] = value
            
            # L2: Disk cache
            self.l2_cache.set(cache_key, compressed_data)
            
            # L3: Redis cache
            if self.redis_client:
                if ttl:
                    await self.redis_client.setex(cache_key, ttl, compressed_data)
                else:
                    await self.redis_client.set(cache_key, compressed_data)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        cache_key = self._generate_key(key)
        
        try:
            # L1: Memory cache
            self.l1_cache.pop(cache_key, None)
            
            # L2: Disk cache
            self.l2_cache.delete(cache_key)
            
            # L3: Redis cache
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache levels"""
        try:
            # L1: Memory cache
            self.l1_cache.clear()
            
            # L2: Disk cache
            self.l2_cache.clear()
            
            # L3: Redis cache
            if self.redis_client:
                await self.redis_client.flushdb()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = sum(self.stats.values())
        hit_rate = 0
        if total_requests > 0:
            hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
            hit_rate = (hits / total_requests) * 100
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_connected": self.redis_client is not None
        }

# Global cache instance
cache = MultiLevelCache()

def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

class ModelCache:
    """Specialized cache for ML models"""
    
    def __init__(self) -> Any:
        self.model_cache = LRUCache(maxsize=10)
        self.pipeline_cache = LRUCache(maxsize=5)
    
    def get_model(self, model_name: str):
        """Get cached model"""
        return self.model_cache.get(model_name)
    
    def set_model(self, model_name: str, model):
        """Cache model"""
        self.model_cache[model_name] = model
    
    def get_pipeline(self, pipeline_name: str):
        """Get cached pipeline"""
        return self.pipeline_cache.get(pipeline_name)
    
    def set_pipeline(self, pipeline_name: str, pipeline):
        """Cache pipeline"""
        self.pipeline_cache[pipeline_name] = pipeline
    
    def clear(self) -> Any:
        """Clear model cache"""
        self.model_cache.clear()
        self.pipeline_cache.clear()

# Global model cache
model_cache = ModelCache()

async def initialize_cache(redis_url: Optional[str] = None):
    """Initialize cache system"""
    global cache
    cache = MultiLevelCache(redis_url=redis_url)
    logger.info("Cache system initialized")

async def cleanup_cache():
    """Cleanup cache system"""
    if cache.redis_client:
        await cache.redis_client.close()
    logger.info("Cache system cleaned up") 