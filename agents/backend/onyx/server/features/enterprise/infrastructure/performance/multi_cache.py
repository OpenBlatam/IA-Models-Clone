from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
        from .ultra_serializer import UltraSerializer
                import redis.asyncio as redis
        import os
        import os
        import os
        import os
            import os
            import shutil
        import os
        import shutil
from typing import Any, List, Dict, Optional
"""
Multi-Level Ultra-Fast Caching
==============================

Ultra-high performance multi-level caching system:
- L1: In-memory cache (fastest, limited size)
- L2: Redis cache (fast, distributed)  
- L3: Disk cache (slower, largest capacity)
- Smart cache invalidation
- Cache warming strategies
- Performance monitoring
"""


logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    avg_access_time: float = 0.0
    hit_ratio: float = 0.0
    
    def update_hit_ratio(self) -> Any:
        """Update hit ratio calculation."""
        total_requests = self.hits + self.misses
        self.hit_ratio = self.hits / max(1, total_requests)


@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> Any:
        """Update access information."""
        self.accessed_at = time.time()
        self.access_count += 1


class ICacheBackend(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class L1MemoryCache(ICacheBackend):
    """Ultra-fast in-memory cache (L1)."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        
    """__init__ function."""
self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheItem] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # Access order tracking for LRU
        self.access_order: List[str] = []
        
        # Frequency tracking for LFU
        self.frequency: Dict[str, int] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache."""
        start_time = time.perf_counter()
        
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if item.is_expired():
                del self.cache[key]
                self._cleanup_tracking(key)
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access tracking
            item.touch()
            self._update_access_tracking(key)
            
            self.stats.hits += 1
            self.stats.update_hit_ratio()
            
            # Update average access time
            access_time = time.perf_counter() - start_time
            total_time = self.stats.avg_access_time * (self.stats.hits + self.stats.misses - 1) + access_time
            self.stats.avg_access_time = total_time / (self.stats.hits + self.stats.misses)
            
            return item.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in L1 cache."""
        with self.lock:
            # Calculate size (approximate)
            try:
                size = len(pickle.dumps(value))
            except:
                size = 100  # Fallback estimate
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl,
                size=size
            )
            
            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_items(1)
            
            # Set item
            self.cache[key] = item
            self._update_access_tracking(key)
            
            self.stats.sets += 1
            self.stats.total_size += size
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete from L1 cache."""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                del self.cache[key]
                self._cleanup_tracking(key)
                
                self.stats.deletes += 1
                self.stats.total_size -= item.size
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear L1 cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency.clear()
            self.stats.total_size = 0
            return True
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for cache strategies."""
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        elif self.strategy == CacheStrategy.LFU:
            self.frequency[key] = self.frequency.get(key, 0) + 1
    
    def _cleanup_tracking(self, key: str):
        """Clean up tracking data for removed key."""
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.frequency:
            del self.frequency[key]
    
    def _evict_items(self, count: int):
        """Evict items based on strategy."""
        for _ in range(count):
            if not self.cache:
                break
            
            if self.strategy == CacheStrategy.LRU:
                if self.access_order:
                    key_to_evict = self.access_order[0]
                else:
                    key_to_evict = next(iter(self.cache))
            elif self.strategy == CacheStrategy.LFU:
                if self.frequency:
                    key_to_evict = min(self.frequency.keys(), key=lambda k: self.frequency[k])
                else:
                    key_to_evict = next(iter(self.cache))
            else:  # FIFO or fallback
                key_to_evict = next(iter(self.cache))
            
            if key_to_evict in self.cache:
                item = self.cache[key_to_evict]
                del self.cache[key_to_evict]
                self._cleanup_tracking(key_to_evict)
                self.stats.evictions += 1
                self.stats.total_size -= item.size
    
    def get_stats(self) -> CacheStats:
        """Get L1 cache statistics."""
        return self.stats


class L2RedisCache(ICacheBackend):
    """Fast distributed Redis cache (L2)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "l2:"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client = None
        self.stats = CacheStats()
        
        # Serializer for Redis storage
        self.serializer = UltraSerializer()
    
    async def _get_redis_client(self) -> Optional[Dict[str, Any]]:
        """Get or create Redis client."""
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
            except ImportError:
                logger.error("redis not installed. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 Redis cache."""
        start_time = time.perf_counter()
        
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            data = await redis_client.get(redis_key)
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            value = await self.serializer.deserialize_async(data)
            
            self.stats.hits += 1
            self.stats.update_hit_ratio()
            
            # Update average access time
            access_time = time.perf_counter() - start_time
            total_time = self.stats.avg_access_time * (self.stats.hits + self.stats.misses - 1) + access_time
            self.stats.avg_access_time = total_time / (self.stats.hits + self.stats.misses)
            
            return value
            
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in L2 Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            # Serialize
            data = await self.serializer.serialize_async(value)
            
            # Set with TTL if specified
            if ttl:
                await redis_client.setex(redis_key, int(ttl), data)
            else:
                await redis_client.set(redis_key, data)
            
            self.stats.sets += 1
            self.stats.total_size += len(data)
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from L2 Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            deleted = await redis_client.delete(redis_key)
            if deleted:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear L2 Redis cache (with prefix)."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get all keys with prefix
            pattern = f"{self.key_prefix}*"
            keys = await redis_client.keys(pattern)
            
            if keys:
                await redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache clear error: {e}")
            return False
    
    def get_stats(self) -> CacheStats:
        """Get L2 cache statistics."""
        return self.stats


class L3DiskCache(ICacheBackend):
    """Disk-based cache for large capacity (L3)."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1000):
        
    """__init__ function."""
self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.stats = CacheStats()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Thread pool for disk operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L3 disk cache."""
        start_time = time.perf_counter()
        
        try:
            file_path = self._get_file_path(key)
            
            # Read from disk in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self.executor, self._read_file, file_path)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            value = pickle.loads(data)
            
            self.stats.hits += 1
            self.stats.update_hit_ratio()
            
            # Update average access time
            access_time = time.perf_counter() - start_time
            total_time = self.stats.avg_access_time * (self.stats.hits + self.stats.misses - 1) + access_time
            self.stats.avg_access_time = total_time / (self.stats.hits + self.stats.misses)
            
            return value
            
        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
            self.stats.misses += 1
            return None
    
    def _read_file(self, file_path: str) -> Optional[bytes]:
        """Read file from disk."""
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in L3 disk cache."""
        try:
            file_path = self._get_file_path(key)
            
            # Serialize
            data = pickle.dumps(value)
            
            # Write to disk in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self.executor, self._write_file, file_path, data)
            
            if success:
                self.stats.sets += 1
                self.stats.total_size += len(data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
            return False
    
    def _write_file(self, file_path: str, data: bytes) -> bool:
        """Write file to disk."""
        try:
            with open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(data)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return True
        except:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from L3 disk cache."""
        try:
            file_path = self._get_file_path(key)
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self.executor, self._delete_file, file_path)
            
            if success:
                self.stats.deletes += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"L3 cache delete error: {e}")
            return False
    
    def _delete_file(self, file_path: str) -> bool:
        """Delete file from disk."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except:
            return False
    
    async def clear(self) -> bool:
        """Clear L3 disk cache."""
        try:
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self.executor, self._clear_directory)
            return success
            
        except Exception as e:
            logger.error(f"L3 cache clear error: {e}")
            return False
    
    def _clear_directory(self) -> bool:
        """Clear cache directory."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            return True
        except:
            return False
    
    def get_stats(self) -> CacheStats:
        """Get L3 cache statistics."""
        return self.stats


class MultiLevelCache:
    """Ultra-fast multi-level cache combining L1, L2, and L3."""
    
    def __init__(self, 
                 l1_cache: Optional[L1MemoryCache] = None,
                 l2_cache: Optional[L2RedisCache] = None,
                 l3_cache: Optional[L3DiskCache] = None):
        
        
    """__init__ function."""
self.l1_cache = l1_cache or L1MemoryCache()
        self.l2_cache = l2_cache
        self.l3_cache = l3_cache
        
        self.total_stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache (L1 -> L2 -> L3)."""
        # Try L1 first (fastest)
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 (Redis)
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                await self.l1_cache.set(key, value)
                return value
        
        # Try L3 (Disk)
        if self.l3_cache:
            value = await self.l3_cache.get(key)
            if value is not None:
                # Promote to L2 and L1
                if self.l2_cache:
                    await self.l2_cache.set(key, value)
                await self.l1_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in all cache levels."""
        results = []
        
        # Set in L1
        results.append(await self.l1_cache.set(key, value, ttl))
        
        # Set in L2
        if self.l2_cache:
            results.append(await self.l2_cache.set(key, value, ttl))
        
        # Set in L3
        if self.l3_cache:
            results.append(await self.l3_cache.set(key, value, ttl))
        
        return any(results)
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        results = []
        
        results.append(await self.l1_cache.delete(key))
        
        if self.l2_cache:
            results.append(await self.l2_cache.delete(key))
        
        if self.l3_cache:
            results.append(await self.l3_cache.delete(key))
        
        return any(results)
    
    async def clear(self) -> bool:
        """Clear all cache levels."""
        results = []
        
        results.append(await self.l1_cache.clear())
        
        if self.l2_cache:
            results.append(await self.l2_cache.clear())
        
        if self.l3_cache:
            results.append(await self.l3_cache.clear())
        
        return all(results)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "l1_stats": self.l1_cache.get_stats().__dict__,
        }
        
        if self.l2_cache:
            stats["l2_stats"] = self.l2_cache.get_stats().__dict__
        
        if self.l3_cache:
            stats["l3_stats"] = self.l3_cache.get_stats().__dict__
        
        # Calculate combined stats
        total_hits = stats["l1_stats"]["hits"]
        total_misses = stats["l1_stats"]["misses"]
        
        if "l2_stats" in stats:
            total_hits += stats["l2_stats"]["hits"]
            total_misses += stats["l2_stats"]["misses"]
        
        if "l3_stats" in stats:
            total_hits += stats["l3_stats"]["hits"]
            total_misses += stats["l3_stats"]["misses"]
        
        stats["combined"] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "combined_hit_ratio": total_hits / max(1, total_hits + total_misses),
            "cache_levels": len([cache for cache in [self.l1_cache, self.l2_cache, self.l3_cache] if cache])
        }
        
        return stats 