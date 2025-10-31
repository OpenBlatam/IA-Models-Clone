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
import json
import pickle
import hashlib
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import aioredis
import orjson
import ujson
import zstandard as zstd
import lz4.frame
import brotli
import numpy as np
import psutil
import gc
from collections import OrderedDict
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import mmap
import tempfile
import os
from pathlib import Path

from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionType(Enum):
    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"
    BROTLI = "brotli"
    GZIP = "gzip"

class CacheLevel(Enum):
    L1 = "l1"  # Memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Disk cache

@dataclass
class CacheConfig:
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_disk_size: int = 1024 * 1024 * 1024   # 1GB
    ttl: int = 3600  # 1 hour
    compression: CompressionType = CompressionType.ZSTD
    compression_level: int = 3
    enable_stats: bool = True
    enable_eviction: bool = True
    eviction_policy: str = "lru"  # lru, lfu, fifo
    batch_size: int = 100
    max_workers: int = 4

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    compression_ratio: float = 1.0
    memory_usage: int = 0
    disk_usage: int = 0

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, maxsize: int = 128):
        
    """__init__ function."""
self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    # Remove least recently used
                    self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        with self.lock:
            return list(self.cache.keys())

class OptimizedCacheManager:
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 config: CacheConfig = None):
        
        
    """__init__ function."""
self.config = config or CacheConfig()
        self.redis_url = redis_url
        
        # Initialize cache levels
        self.l1_cache = LRUCache(maxsize=1000)  # Memory cache
        self.l2_cache = None  # Redis cache
        self.l3_cache_path = Path(tempfile.gettempdir()) / "optimized_cache"
        self.l3_cache_path.mkdir(exist_ok=True)
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = threading.Lock()
        
        # Compression
        self.compressor = self._get_compressor()
        
        # Initialize Redis connection
        self._init_redis()
        
        logger.info(f"OptimizedCacheManager initialized with {self.config.compression.value} compression")

    def _init_redis(self) -> Any:
        """Initialize Redis connection"""
        try:
            self.l2_cache = redis.from_url(self.redis_url, decode_responses=False)
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using L1 and L3 cache only.")
            self.l2_cache = None

    def _get_compressor(self) -> Callable:
        """Get compression function based on config"""
        if self.config.compression == CompressionType.ZSTD:
            return lambda data: zstd.compress(data, level=self.config.compression_level)
        elif self.config.compression == CompressionType.LZ4:
            return lambda data: lz4.frame.compress(data, compression_level=self.config.compression_level)
        elif self.config.compression == CompressionType.BROTLI:
            return lambda data: brotli.compress(data, quality=self.config.compression_level)
        else:
            return lambda data: data

    def _get_decompressor(self) -> Callable:
        """Get decompression function based on config"""
        if self.config.compression == CompressionType.ZSTD:
            return lambda data: zstd.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            return lambda data: lz4.frame.decompress(data)
        elif self.config.compression == CompressionType.BROTLI:
            return lambda data: brotli.decompress(data)
        else:
            return lambda data: data

    def _generate_key(self, key: str) -> str:
        """Generate cache key with hash"""
        return hashlib.sha256(key.encode()).hexdigest()

    def _serialize(self, value: Any) -> bytes:
        """Serialize value with compression"""
        try:
            # Try JSON first for simple types
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                data = orjson.dumps(value)
            else:
                # Use pickle for complex objects
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress
            compressed = self.compressor(data)
            
            # Update compression ratio
            if len(data) > 0:
                ratio = len(compressed) / len(data)
                with self.stats_lock:
                    self.stats.compression_ratio = (self.stats.compression_ratio + ratio) / 2
            
            return compressed
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value with decompression"""
        try:
            # Decompress
            decompressed = self._get_decompressor()(data)
            
            # Try JSON first
            try:
                return orjson.loads(decompressed)
            except:
                # Fallback to pickle
                return pickle.loads(decompressed)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        cache_key = self._generate_key(key)
        
        # L1 Cache (Memory)
        value = self.l1_cache.get(cache_key)
        if value is not None:
            with self.stats_lock:
                self.stats.hits += 1
            return value
        
        # L2 Cache (Redis)
        if self.l2_cache:
            try:
                value = await self.l2_cache.get(cache_key)
                if value is not None:
                    deserialized = self._deserialize(value)
                    # Store in L1 cache
                    self.l1_cache.set(cache_key, deserialized)
                    with self.stats_lock:
                        self.stats.hits += 1
                    return deserialized
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # L3 Cache (Disk)
        try:
            value = await self._get_from_disk(cache_key)
            if value is not None:
                # Store in L1 and L2 cache
                self.l1_cache.set(cache_key, value)
                if self.l2_cache:
                    await self._set_to_redis(cache_key, value)
                with self.stats_lock:
                    self.stats.hits += 1
                return value
        except Exception as e:
            logger.error(f"Disk get error: {e}")
        
        # Cache miss
        with self.stats_lock:
            self.stats.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in multi-level cache"""
        cache_key = self._generate_key(key)
        ttl = ttl or self.config.ttl
        
        try:
            # L1 Cache (Memory)
            self.l1_cache.set(cache_key, value)
            
            # L2 Cache (Redis) - async
            if self.l2_cache:
                await self._set_to_redis(cache_key, value, ttl)
            
            # L3 Cache (Disk) - async
            await self._set_to_disk(cache_key, value)
            
            with self.stats_lock:
                self.stats.sets += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        cache_key = self._generate_key(key)
        
        try:
            # L1 Cache
            self.l1_cache.delete(cache_key)
            
            # L2 Cache
            if self.l2_cache:
                await self.l2_cache.delete(cache_key)
            
            # L3 Cache
            await self._delete_from_disk(cache_key)
            
            with self.stats_lock:
                self.stats.deletes += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def _set_to_redis(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value to Redis cache"""
        if not self.l2_cache:
            return
        
        try:
            serialized = self._serialize(value)
            await self.l2_cache.set(key, serialized, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        file_path = self.l3_cache_path / f"{key}.cache"
        
        if not file_path.exists():
            return None
        
        try:
            # Check if file is expired
            if time.time() - file_path.stat().st_mtime > self.config.ttl:
                file_path.unlink()
                return None
            
            # Read file
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            return self._deserialize(data)
            
        except Exception as e:
            logger.error(f"Disk read error: {e}")
            return None

    async def _set_to_disk(self, key: str, value: Any) -> None:
        """Set value to disk cache"""
        try:
            # Check disk space
            if self._get_disk_usage() > self.config.max_disk_size:
                await self._cleanup_disk_cache()
            
            file_path = self.l3_cache_path / f"{key}.cache"
            serialized = self._serialize(value)
            
            with open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(serialized)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Update disk usage stats
            with self.stats_lock:
                self.stats.disk_usage = self._get_disk_usage()
                
        except Exception as e:
            logger.error(f"Disk write error: {e}")

    async def _delete_from_disk(self, key: str) -> None:
        """Delete value from disk cache"""
        try:
            file_path = self.l3_cache_path / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Disk delete error: {e}")

    def _get_disk_usage(self) -> int:
        """Get current disk usage"""
        try:
            total_size = 0
            for file_path in self.l3_cache_path.glob("*.cache"):
                total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0

    async def _cleanup_disk_cache(self) -> None:
        """Clean up disk cache based on eviction policy"""
        try:
            files = []
            for file_path in self.l3_cache_path.glob("*.cache"):
                stat = file_path.stat()
                files.append((file_path, stat.st_mtime, stat.st_size))
            
            if self.config.eviction_policy == "lru":
                # Sort by modification time (oldest first)
                files.sort(key=lambda x: x[1])
            elif self.config.eviction_policy == "lfu":
                # For simplicity, use LRU as LFU requires access counting
                files.sort(key=lambda x: x[1])
            else:  # fifo
                # Sort by creation time
                files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            current_size = sum(size for _, _, size in files)
            for file_path, _, size in files:
                if current_size <= self.config.max_disk_size * 0.8:  # Keep 20% buffer
                    break
                file_path.unlink()
                current_size -= size
                with self.stats_lock:
                    self.stats.evictions += 1
                    
        except Exception as e:
            logger.error(f"Disk cleanup error: {e}")

    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values in batch"""
        results = {}
        
        # Process in batches
        for i in range(0, len(keys), self.config.batch_size):
            batch_keys = keys[i:i + self.config.batch_size]
            
            # Create tasks for batch
            tasks = [self.get(key) for key in batch_keys]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for key, result in zip(batch_keys, batch_results):
                if not isinstance(result, Exception):
                    results[key] = result
        
        return results

    async def batch_set(self, items: Dict[str, Any], ttl: int = None) -> Dict[str, bool]:
        """Set multiple values in batch"""
        results = {}
        
        # Process in batches
        items_list = list(items.items())
        for i in range(0, len(items_list), self.config.batch_size):
            batch_items = items_list[i:i + self.config.batch_size]
            
            # Create tasks for batch
            tasks = [self.set(key, value, ttl) for key, value in batch_items]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for (key, _), result in zip(batch_items, batch_results):
                results[key] = not isinstance(result, Exception)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.stats_lock:
            stats = asdict(self.stats)
            stats['memory_usage'] = self._get_memory_usage()
            stats['disk_usage'] = self._get_disk_usage()
            stats['l1_size'] = self.l1_cache.size()
            stats['hit_rate'] = self.stats.hits / (self.stats.hits + self.stats.misses) if (self.stats.hits + self.stats.misses) > 0 else 0
            return stats

    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        try:
            # Estimate memory usage of L1 cache
            total_size = 0
            for key, value in self.l1_cache.cache.items():
                total_size += len(key.encode()) + len(str(value).encode())
            return total_size
        except Exception:
            return 0

    async def clear(self) -> None:
        """Clear all cache levels"""
        try:
            # L1 Cache
            self.l1_cache.clear()
            
            # L2 Cache
            if self.l2_cache:
                await self.l2_cache.flushdb()
            
            # L3 Cache
            for file_path in self.l3_cache_path.glob("*.cache"):
                file_path.unlink()
            
            # Reset stats
            with self.stats_lock:
                self.stats = CacheStats()
            
            logger.info("All cache levels cleared")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    async def close(self) -> None:
        """Close cache manager and cleanup resources"""
        try:
            self.thread_pool.shutdown(wait=True)
            
            if self.l2_cache:
                await self.l2_cache.close()
            
            # Cleanup disk cache
            await self._cleanup_disk_cache()
            
            logger.info("OptimizedCacheManager closed")
            
        except Exception as e:
            logger.error(f"Cache close error: {e}")

# Cache decorator
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager:
                cached_result = await cache_manager.get(key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            if cache_manager:
                await cache_manager.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage example
async def main():
    
    """main function."""
# Initialize cache manager
    cache_manager = OptimizedCacheManager(
        redis_url="redis://localhost:6379",
        config=CacheConfig(
            max_memory_size=50 * 1024 * 1024,  # 50MB
            max_disk_size=500 * 1024 * 1024,   # 500MB
            ttl=1800,  # 30 minutes
            compression=CompressionType.ZSTD,
            compression_level=3,
            enable_stats=True,
            enable_eviction=True,
            eviction_policy="lru"
        )
    )
    
    try:
        # Set values
        await cache_manager.set("user:123", {"name": "John", "age": 30})
        await cache_manager.set("config:app", {"debug": True, "version": "1.0.0"})
        
        # Get values
        user = await cache_manager.get("user:123")
        config = await cache_manager.get("config:app")
        
        print(f"User: {user}")
        print(f"Config: {config}")
        
        # Batch operations
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        await cache_manager.batch_set(items)
        results = await cache_manager.batch_get(["key1", "key2", "key3"])
        
        print(f"Batch results: {results}")
        
        # Get statistics
        stats = cache_manager.get_stats()
        print(f"Cache stats: {stats}")
        
    finally:
        await cache_manager.close()

match __name__:
    case "__main__":
    asyncio.run(main()) 