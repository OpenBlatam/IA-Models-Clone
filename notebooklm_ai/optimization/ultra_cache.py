from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import orjson
import lz4.frame
import aioredis
import structlog
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Ultra-Fast Multi-Level Caching System
⚡ L1 Memory + L2 Redis with compression and serialization optimization
"""


logger = structlog.get_logger()

@dataclass
class CacheConfig:
    """Cache configuration."""
    l1_size: int = 10000
    l2_ttl: int = 3600
    enable_compression: bool = True
    compression_level: int = 6

class L1MemoryCache:
    """Ultra-fast L1 memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10000):
        
    """__init__ function."""
self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache."""
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            self.stats["hits"] += 1
            return self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Set value in L1 cache."""
        if key in self.cache:
            self.access_order.remove(key)
        else:
            if len(self.cache) >= self.max_size:
                evicted_key = self.access_order.popleft()
                del self.cache[evicted_key]
                self.stats["evictions"] += 1
        
        self.cache[key] = value
        self.access_order.append(key)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate
        }

class L2RedisCache:
    """Ultra-fast L2 Redis cache with connection pooling."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis = None
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
    
    async def connect(self) -> Any:
        """Connect to Redis with connection pooling."""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        try:
            await self.connect()
            value = await self.redis.get(key)
            if value:
                self.stats["hits"] += 1
                return value
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Redis get error", error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in L2 cache."""
        try:
            await self.connect()
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Redis set error", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return dict(self.stats)

class UltraCache:
    """Ultra-fast multi-level cache with compression."""
    
    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.config = config
        self.l1_cache = L1MemoryCache(config.l1_size)
        self.l2_cache = L2RedisCache(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 cache first (fastest)
        l1_value = await self.l1_cache.get(key)
        if l1_value is not None:
            return l1_value
        
        # Try L2 cache
        l2_value = await self.l2_cache.get(key)
        if l2_value is not None:
            # Decompress and deserialize
            if self.config.enable_compression:
                l2_value = lz4.frame.decompress(l2_value.encode())
            else:
                l2_value = l2_value.encode()
            
            deserialized = orjson.loads(l2_value)
            
            # Store in L1 cache for next time
            await self.l1_cache.set(key, deserialized)
            
            return deserialized
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in multi-level cache."""
        ttl = ttl or self.config.l2_ttl
        
        # Serialize and compress
        serialized = orjson.dumps(value)
        if self.config.enable_compression:
            compressed = lz4.frame.compress(serialized, compression_level=self.config.compression_level)
            cache_value = compressed.decode('latin1')
        else:
            cache_value = serialized.decode('utf-8')
        
        # Store in both caches
        l1_success = await self.l1_cache.set(key, value)
        l2_success = await self.l2_cache.set(key, cache_value, ttl)
        
        return l1_success and l2_success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "l1_cache": self.l1_cache.get_stats(),
            "l2_cache": self.l2_cache.get_stats()
        }

# Global cache instance
_cache = None

def get_cache(config: CacheConfig = None, redis_url: str = "redis://localhost:6379") -> UltraCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = UltraCache(config or CacheConfig(), redis_url)
    return _cache 