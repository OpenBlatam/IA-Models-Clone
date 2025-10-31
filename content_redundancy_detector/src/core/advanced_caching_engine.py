"""
Advanced Caching Engine - Multi-level caching with intelligent strategies
"""

import asyncio
import json
import logging
import pickle
import time
import zlib
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Caching libraries
import redis.asyncio as redis
from cachetools import TTLCache, LRUCache, LFUCache, RRCache
import diskcache

# Compression
import lz4.frame
import brotli

# Serialization
import msgpack

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    RR = "rr"  # Random replacement
    ADAPTIVE = "adaptive"


class SerializationFormat(Enum):
    """Serialization format types"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"


@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 1000
    ttl_seconds: int = 3600
    strategy: CacheStrategy = CacheStrategy.LRU
    serialization: SerializationFormat = SerializationFormat.JSON
    compression: bool = False
    compression_level: int = 6
    enable_redis: bool = True
    enable_disk: bool = False
    disk_path: str = "./cache"
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    compressed: bool = False


class MemoryCache:
    """Advanced in-memory cache with multiple strategies"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        # Initialize cache based on strategy
        if config.strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.LFU:
            self.cache = LFUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.TTL:
            self.cache = TTLCache(maxsize=config.max_size, ttl=config.ttl_seconds)
        elif config.strategy == CacheStrategy.RR:
            self.cache = RRCache(maxsize=config.max_size)
        else:
            self.cache = LRUCache(maxsize=config.max_size)  # Default
        
        # Hook into cache events
        self._setup_cache_hooks()
    
    def _setup_cache_hooks(self):
        """Setup cache event hooks"""
        original_popitem = self.cache.popitem
        original_pop = self.cache.pop
        
        def hooked_popitem(*args, **kwargs):
            with self._lock:
                self.stats.evictions += 1
            return original_popitem(*args, **kwargs)
        
        def hooked_pop(*args, **kwargs):
            with self._lock:
                if args[0] in self.cache:
                    self.stats.evictions += 1
            return original_pop(*args, **kwargs)
        
        self.cache.popitem = hooked_popitem
        self.cache.pop = hooked_pop
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self.cache:
                self.stats.hits += 1
                self._update_stats()
                return self.cache[key]
            else:
                self.stats.misses += 1
                self._update_stats()
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            try:
                # Serialize value
                serialized_value = self._serialize(value)
                
                # Compress if enabled
                if self.config.compression:
                    serialized_value = self._compress(serialized_value)
                
                # Calculate size
                size_bytes = len(serialized_value)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=serialized_value,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl or self.config.ttl_seconds),
                    size_bytes=size_bytes,
                    compressed=self.config.compression
                )
                
                self.cache[key] = entry
                self.stats.sets += 1
                self._update_stats()
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache value for key '{key}': {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value based on format"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.dumps(value, default=str).encode('utf-8')
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(value)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            return msgpack.packb(value, default=str)
        else:
            return json.dumps(value, default=str).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value based on format"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            return msgpack.unpackb(data, raw=False)
        else:
            return json.loads(data.decode('utf-8'))
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data"""
        if self.config.compression_level == 1:
            return lz4.frame.compress(data)
        else:
            return brotli.compress(data, quality=self.config.compression_level)
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        try:
            return lz4.frame.decompress(data)
        except:
            return brotli.decompress(data)
    
    def _update_stats(self):
        """Update cache statistics"""
        self.stats.size = len(self.cache)
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = (self.stats.hits / total_requests) * 100
            self.stats.miss_rate = (self.stats.misses / total_requests) * 100
        
        # Calculate memory usage (rough estimate)
        self.stats.memory_usage_mb = sum(
            entry.size_bytes for entry in self.cache.values() 
            if isinstance(entry, CacheEntry)
        ) / 1024 / 1024
        
        self.stats.last_updated = datetime.now()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self._update_stats()
            return self.stats


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            with self._lock:
                data = await self.redis_client.get(key)
                if data:
                    self.stats.hits += 1
                    self._update_stats()
                    return self._deserialize(data)
                else:
                    self.stats.misses += 1
                    self._update_stats()
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting value from Redis for key '{key}': {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            with self._lock:
                # Serialize value
                serialized_value = self._serialize(value)
                
                # Compress if enabled
                if self.config.compression:
                    serialized_value = self._compress(serialized_value)
                
                # Set in Redis with TTL
                ttl_seconds = ttl or self.config.ttl_seconds
                await self.redis_client.setex(key, ttl_seconds, serialized_value)
                
                self.stats.sets += 1
                self._update_stats()
                return True
                
        except Exception as e:
            logger.error(f"Error setting value in Redis for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            with self._lock:
                result = await self.redis_client.delete(key)
                if result:
                    self.stats.deletes += 1
                    self._update_stats()
                return bool(result)
                
        except Exception as e:
            logger.error(f"Error deleting value from Redis for key '{key}': {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        if not self.redis_client:
            return
        
        try:
            with self._lock:
                await self.redis_client.flushdb()
                self.stats = CacheStats()
                
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value based on format"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.dumps(value, default=str).encode('utf-8')
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(value)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            return msgpack.packb(value, default=str)
        else:
            return json.dumps(value, default=str).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value based on format"""
        if self.config.serialization == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            return msgpack.unpackb(data, raw=False)
        else:
            return json.loads(data.decode('utf-8'))
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data"""
        if self.config.compression_level == 1:
            return lz4.frame.compress(data)
        else:
            return brotli.compress(data, quality=self.config.compression_level)
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        try:
            return lz4.frame.decompress(data)
        except:
            return brotli.decompress(data)
    
    def _update_stats(self):
        """Update cache statistics"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = (self.stats.hits / total_requests) * 100
            self.stats.miss_rate = (self.stats.misses / total_requests) * 100
        
        self.stats.last_updated = datetime.now()
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        if not self.redis_client:
            return self.stats
        
        try:
            with self._lock:
                info = await self.redis_client.info('memory')
                self.stats.memory_usage_mb = info.get('used_memory', 0) / 1024 / 1024
                self._update_stats()
                return self.stats
                
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return self.stats


class DiskCache:
    """Disk-based persistent cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = diskcache.Cache(config.disk_path)
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        with self._lock:
            try:
                if key in self.cache:
                    self.stats.hits += 1
                    self._update_stats()
                    return self.cache[key]
                else:
                    self.stats.misses += 1
                    self._update_stats()
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting value from disk cache for key '{key}': {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache"""
        with self._lock:
            try:
                ttl_seconds = ttl or self.config.ttl_seconds
                self.cache.set(key, value, expire=ttl_seconds)
                self.stats.sets += 1
                self._update_stats()
                return True
                
            except Exception as e:
                logger.error(f"Error setting value in disk cache for key '{key}': {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from disk cache"""
        with self._lock:
            try:
                if key in self.cache:
                    del self.cache[key]
                    self.stats.deletes += 1
                    self._update_stats()
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Error deleting value from disk cache for key '{key}': {e}")
                return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            try:
                self.cache.clear()
                self.stats = CacheStats()
                
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
    
    def _update_stats(self):
        """Update cache statistics"""
        self.stats.size = len(self.cache)
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = (self.stats.hits / total_requests) * 100
            self.stats.miss_rate = (self.stats.misses / total_requests) * 100
        
        self.stats.last_updated = datetime.now()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self._update_stats()
            return self.stats


class AdvancedCachingEngine:
    """Advanced multi-level caching engine"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config) if config.enable_redis else None
        self.disk_cache = DiskCache(config) if config.enable_disk else None
        
        self.cache_hierarchy = []
        if self.memory_cache:
            self.cache_hierarchy.append(("memory", self.memory_cache))
        if self.redis_cache:
            self.cache_hierarchy.append(("redis", self.redis_cache))
        if self.disk_cache:
            self.cache_hierarchy.append(("disk", self.disk_cache))
        
        self._connected = False
    
    async def initialize(self) -> bool:
        """Initialize caching engine"""
        try:
            # Connect to Redis if enabled
            if self.redis_cache:
                redis_connected = await self.redis_cache.connect()
                if not redis_connected:
                    logger.warning("Redis connection failed, continuing without Redis cache")
                    self.redis_cache = None
                    self.cache_hierarchy = [(name, cache) for name, cache in self.cache_hierarchy if name != "redis"]
            
            self._connected = True
            logger.info("Advanced Caching Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Caching Engine: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown caching engine"""
        try:
            if self.redis_cache:
                await self.redis_cache.disconnect()
            
            self._connected = False
            logger.info("Advanced Caching Engine shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down Advanced Caching Engine: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        if not self._connected:
            return None
        
        # Try each cache level in order
        for cache_name, cache in self.cache_hierarchy:
            try:
                if cache_name == "redis":
                    value = await cache.get(key)
                else:
                    value = cache.get(key)
                
                if value is not None:
                    # Promote to higher cache levels
                    await self._promote_to_higher_levels(key, value, cache_name)
                    return value
                    
            except Exception as e:
                logger.error(f"Error getting value from {cache_name} cache: {e}")
                continue
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache hierarchy"""
        if not self._connected:
            return False
        
        success = False
        
        # Set in all cache levels
        for cache_name, cache in self.cache_hierarchy:
            try:
                if cache_name == "redis":
                    result = await cache.set(key, value, ttl)
                else:
                    result = cache.set(key, value, ttl)
                
                if result:
                    success = True
                    
            except Exception as e:
                logger.error(f"Error setting value in {cache_name} cache: {e}")
                continue
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache hierarchy"""
        if not self._connected:
            return False
        
        success = False
        
        # Delete from all cache levels
        for cache_name, cache in self.cache_hierarchy:
            try:
                if cache_name == "redis":
                    result = await cache.delete(key)
                else:
                    result = cache.delete(key)
                
                if result:
                    success = True
                    
            except Exception as e:
                logger.error(f"Error deleting value from {cache_name} cache: {e}")
                continue
        
        return success
    
    async def clear(self) -> None:
        """Clear all cache levels"""
        if not self._connected:
            return
        
        for cache_name, cache in self.cache_hierarchy:
            try:
                if cache_name == "redis":
                    await cache.clear()
                else:
                    cache.clear()
                    
            except Exception as e:
                logger.error(f"Error clearing {cache_name} cache: {e}")
    
    async def _promote_to_higher_levels(self, key: str, value: Any, current_level: str) -> None:
        """Promote value to higher cache levels"""
        current_index = next(i for i, (name, _) in enumerate(self.cache_hierarchy) if name == current_level)
        
        # Promote to higher levels (lower indices)
        for i in range(current_index):
            cache_name, cache = self.cache_hierarchy[i]
            try:
                if cache_name == "redis":
                    await cache.set(key, value)
                else:
                    cache.set(key, value)
                    
            except Exception as e:
                logger.error(f"Error promoting value to {cache_name} cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self._connected:
            return {"error": "Cache engine not connected"}
        
        stats = {
            "timestamp": datetime.now(),
            "cache_levels": {},
            "overall_stats": {
                "total_hits": 0,
                "total_misses": 0,
                "total_sets": 0,
                "total_deletes": 0,
                "total_evictions": 0,
                "overall_hit_rate": 0.0
            }
        }
        
        # Get stats from each cache level
        for cache_name, cache in self.cache_hierarchy:
            try:
                if cache_name == "redis":
                    cache_stats = await cache.get_stats()
                else:
                    cache_stats = cache.get_stats()
                
                stats["cache_levels"][cache_name] = {
                    "hits": cache_stats.hits,
                    "misses": cache_stats.misses,
                    "sets": cache_stats.sets,
                    "deletes": cache_stats.deletes,
                    "evictions": cache_stats.evictions,
                    "size": cache_stats.size,
                    "memory_usage_mb": cache_stats.memory_usage_mb,
                    "hit_rate": cache_stats.hit_rate,
                    "miss_rate": cache_stats.miss_rate,
                    "last_updated": cache_stats.last_updated
                }
                
                # Aggregate overall stats
                stats["overall_stats"]["total_hits"] += cache_stats.hits
                stats["overall_stats"]["total_misses"] += cache_stats.misses
                stats["overall_stats"]["total_sets"] += cache_stats.sets
                stats["overall_stats"]["total_deletes"] += cache_stats.deletes
                stats["overall_stats"]["total_evictions"] += cache_stats.evictions
                
            except Exception as e:
                logger.error(f"Error getting stats from {cache_name} cache: {e}")
                stats["cache_levels"][cache_name] = {"error": str(e)}
        
        # Calculate overall hit rate
        total_requests = stats["overall_stats"]["total_hits"] + stats["overall_stats"]["total_misses"]
        if total_requests > 0:
            stats["overall_stats"]["overall_hit_rate"] = (
                stats["overall_stats"]["total_hits"] / total_requests * 100
            )
        
        return stats
    
    async def optimize_cache_strategy(self) -> Dict[str, Any]:
        """Optimize cache strategy based on usage patterns"""
        if not self._connected:
            return {"error": "Cache engine not connected"}
        
        start_time = time.time()
        
        # Get current stats
        stats = await self.get_stats()
        
        optimizations = []
        
        # Analyze hit rates
        for cache_name, cache_stats in stats["cache_levels"].items():
            if "hit_rate" in cache_stats:
                hit_rate = cache_stats["hit_rate"]
                
                if hit_rate < 50:
                    optimizations.append({
                        "cache_level": cache_name,
                        "issue": "Low hit rate",
                        "current_hit_rate": hit_rate,
                        "recommendation": "Consider increasing cache size or TTL"
                    })
                elif hit_rate > 90:
                    optimizations.append({
                        "cache_level": cache_name,
                        "issue": "Very high hit rate",
                        "current_hit_rate": hit_rate,
                        "recommendation": "Cache is working well, consider reducing TTL to save memory"
                    })
        
        # Analyze memory usage
        for cache_name, cache_stats in stats["cache_levels"].items():
            if "memory_usage_mb" in cache_stats:
                memory_usage = cache_stats["memory_usage_mb"]
                
                if memory_usage > 100:  # 100MB threshold
                    optimizations.append({
                        "cache_level": cache_name,
                        "issue": "High memory usage",
                        "current_usage_mb": memory_usage,
                        "recommendation": "Consider reducing cache size or enabling compression"
                    })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "current_stats": stats,
            "optimizations": optimizations,
            "execution_time_ms": execution_time
        }
    
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        # Combine all arguments
        key_parts = [prefix] + [str(arg) for arg in args]
        
        # Add keyword arguments sorted by key
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        # Join and hash
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def cache_function_result(
        self, 
        func: Callable, 
        key: str, 
        ttl: Optional[int] = None,
        *args, 
        **kwargs
    ) -> Any:
        """Cache function result"""
        # Try to get from cache first
        cached_result = await self.get(key)
        if cached_result is not None:
            return cached_result
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        # Cache result
        await self.set(key, result, ttl)
        
        return result


# Global instance
advanced_caching_engine: Optional[AdvancedCachingEngine] = None


async def initialize_advanced_caching_engine(config: Optional[CacheConfig] = None) -> None:
    """Initialize advanced caching engine"""
    global advanced_caching_engine
    
    if config is None:
        config = CacheConfig()
    
    advanced_caching_engine = AdvancedCachingEngine(config)
    success = await advanced_caching_engine.initialize()
    
    if success:
        logger.info("Advanced Caching Engine initialized successfully")
    else:
        logger.error("Failed to initialize Advanced Caching Engine")


async def get_advanced_caching_engine() -> Optional[AdvancedCachingEngine]:
    """Get advanced caching engine instance"""
    return advanced_caching_engine