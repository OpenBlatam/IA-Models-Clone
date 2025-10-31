"""
Refactored Cache System

Sistema de caché y persistencia refactorizado para el AI History Comparison System.
Maneja caché multi-nivel, persistencia inteligente, invalidación automática y optimización.
"""

import asyncio
import logging
import json
import pickle
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import weakref
from collections import OrderedDict
import os
import sqlite3
import aiofiles

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1 = "l1"  # Memory cache
    L2 = "l2"  # Disk cache
    L3 = "l3"  # Database cache
    L4 = "l4"  # Distributed cache


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class CacheOperation(Enum):
    """Cache operation enumeration"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"
    INVALIDATE = "invalidate"
    REFRESH = "refresh"


@dataclass
class CacheMetadata:
    """Cache metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[timedelta] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    is_persistent: bool = False
    compression_ratio: float = 1.0


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size"""
        pass


class MemoryCacheBackend(CacheBackend):
    """Memory cache backend with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self._cache: OrderedDict[str, CacheMetadata] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key in self._cache:
                metadata = self._cache[key]
                
                # Check TTL
                if metadata.ttl and datetime.utcnow() - metadata.created_at > metadata.ttl:
                    del self._cache[key]
                    self._current_memory -= metadata.size_bytes
                    self._stats.misses += 1
                    return None
                
                # Update access info
                metadata.accessed_at = datetime.utcnow()
                metadata.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._stats.hits += 1
                return metadata.value
            
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Remove existing key if present
                if key in self._cache:
                    old_metadata = self._cache[key]
                    self._current_memory -= old_metadata.size_bytes
                    del self._cache[key]
                
                # Check memory limit
                if self._current_memory + size_bytes > self._max_memory_bytes:
                    await self._evict_oldest()
                
                # Check size limit
                if len(self._cache) >= self._max_size:
                    await self._evict_oldest()
                
                # Add new entry
                metadata = CacheMetadata(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    accessed_at=datetime.utcnow(),
                    size_bytes=size_bytes,
                    ttl=ttl
                )
                
                self._cache[key] = metadata
                self._current_memory += size_bytes
                self._stats.sets += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        async with self._lock:
            if key in self._cache:
                metadata = self._cache[key]
                del self._cache[key]
                self._current_memory -= metadata.size_bytes
                self._stats.deletes += 1
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all memory cache"""
        async with self._lock:
            self._cache.clear()
            self._current_memory = 0
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        async with self._lock:
            return key in self._cache
    
    async def size(self) -> int:
        """Get memory cache size"""
        async with self._lock:
            return len(self._cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size
    
    async def _evict_oldest(self) -> None:
        """Evict oldest entry"""
        if self._cache:
            key, metadata = self._cache.popitem(last=False)
            self._current_memory -= metadata.size_bytes
            self._stats.evictions += 1
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        total_requests = self._stats.hits + self._stats.misses
        if total_requests > 0:
            self._stats.hit_rate = self._stats.hits / total_requests
            self._stats.miss_rate = self._stats.misses / total_requests
        
        self._stats.total_size = self._current_memory
        return self._stats


class DiskCacheBackend(CacheBackend):
    """Disk cache backend with file storage"""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 1000):
        self._cache_dir = cache_dir
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size = 0
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        async with self._lock:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                self._stats.misses += 1
                return None
            
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    data = await f.read()
                
                # Deserialize
                cache_data = pickle.loads(data)
                
                # Check TTL
                if cache_data.get('ttl') and datetime.utcnow() - cache_data['created_at'] > cache_data['ttl']:
                    await self.delete(key)
                    self._stats.misses += 1
                    return None
                
                self._stats.hits += 1
                return cache_data['value']
                
            except Exception as e:
                logger.error(f"Error reading cache file {file_path}: {e}")
                self._stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in disk cache"""
        async with self._lock:
            try:
                file_path = self._get_file_path(key)
                
                # Serialize data
                cache_data = {
                    'value': value,
                    'created_at': datetime.utcnow(),
                    'ttl': ttl
                }
                
                data = pickle.dumps(cache_data)
                file_size = len(data)
                
                # Check size limit
                if self._current_size + file_size > self._max_size_bytes:
                    await self._cleanup_oldest()
                
                # Write to file
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(data)
                
                self._current_size += file_size
                self._stats.sets += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error writing cache file {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache"""
        async with self._lock:
            file_path = self._get_file_path(key)
            
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self._current_size -= file_size
                    self._stats.deletes += 1
                    return True
                except Exception as e:
                    logger.error(f"Error deleting cache file {file_path}: {e}")
            
            return False
    
    async def clear(self) -> bool:
        """Clear all disk cache"""
        async with self._lock:
            try:
                for filename in os.listdir(self._cache_dir):
                    file_path = os.path.join(self._cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                
                self._current_size = 0
                return True
                
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in disk cache"""
        file_path = self._get_file_path(key)
        return os.path.exists(file_path)
    
    async def size(self) -> int:
        """Get disk cache size"""
        async with self._lock:
            return self._current_size
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self._cache_dir, f"{key_hash}.cache")
    
    async def _cleanup_oldest(self) -> None:
        """Cleanup oldest files"""
        try:
            files = []
            for filename in os.listdir(self._cache_dir):
                file_path = os.path.join(self._cache_dir, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append((file_path, stat.st_mtime))
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            for file_path, _ in files:
                if self._current_size <= self._max_size_bytes * 0.8:  # 80% of limit
                    break
                
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                self._current_size -= file_size
                self._stats.evictions += 1
                
        except Exception as e:
            logger.error(f"Error cleaning up disk cache: {e}")


class DatabaseCacheBackend(CacheBackend):
    """Database cache backend with SQLite"""
    
    def __init__(self, db_path: str = "cache.db"):
        self._db_path = db_path
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at TIMESTAMP,
                accessed_at TIMESTAMP,
                ttl TIMESTAMP,
                size_bytes INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache(accessed_at)
        ''')
        
        conn.commit()
        conn.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from database cache"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT value, created_at, ttl FROM cache WHERE key = ?
                ''', (key,))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    self._stats.misses += 1
                    return None
                
                value, created_at, ttl = row
                created_at = datetime.fromisoformat(created_at)
                
                # Check TTL
                if ttl:
                    ttl_dt = datetime.fromisoformat(ttl)
                    if datetime.utcnow() > ttl_dt:
                        await self.delete(key)
                        self._stats.misses += 1
                        return None
                
                # Update access time
                await self._update_access_time(key)
                
                # Deserialize
                deserialized_value = pickle.loads(value)
                
                self._stats.hits += 1
                return deserialized_value
                
            except Exception as e:
                logger.error(f"Error reading from database cache: {e}")
                self._stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in database cache"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                # Serialize value
                serialized_value = pickle.dumps(value)
                size_bytes = len(serialized_value)
                
                # Calculate TTL
                ttl_timestamp = None
                if ttl:
                    ttl_timestamp = (datetime.utcnow() + ttl).isoformat()
                
                # Insert or update
                cursor.execute('''
                    INSERT OR REPLACE INTO cache 
                    (key, value, created_at, accessed_at, ttl, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    key,
                    serialized_value,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                    ttl_timestamp,
                    size_bytes
                ))
                
                conn.commit()
                conn.close()
                
                self._stats.sets += 1
                return True
                
            except Exception as e:
                logger.error(f"Error writing to database cache: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from database cache"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                deleted = cursor.rowcount > 0
                
                conn.commit()
                conn.close()
                
                if deleted:
                    self._stats.deletes += 1
                
                return deleted
                
            except Exception as e:
                logger.error(f"Error deleting from database cache: {e}")
                return False
    
    async def clear(self) -> bool:
        """Clear all database cache"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM cache')
                
                conn.commit()
                conn.close()
                
                return True
                
            except Exception as e:
                logger.error(f"Error clearing database cache: {e}")
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in database cache"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT 1 FROM cache WHERE key = ?', (key,))
                exists = cursor.fetchone() is not None
                
                conn.close()
                
                return exists
                
            except Exception as e:
                logger.error(f"Error checking database cache: {e}")
                return False
    
    async def size(self) -> int:
        """Get database cache size"""
        async with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM cache')
                count = cursor.fetchone()[0]
                
                conn.close()
                
                return count
                
            except Exception as e:
                logger.error(f"Error getting database cache size: {e}")
                return 0
    
    async def _update_access_time(self, key: str) -> None:
        """Update access time for key"""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE cache SET accessed_at = ? WHERE key = ?
            ''', (datetime.utcnow().isoformat(), key))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating access time: {e}")


class RefactoredCacheManager:
    """Refactored cache manager with multi-level caching"""
    
    def __init__(self):
        self._backends: Dict[CacheLevel, CacheBackend] = {}
        self._cache_order: List[CacheLevel] = [
            CacheLevel.L1,  # Memory first
            CacheLevel.L2,  # Disk second
            CacheLevel.L3   # Database third
        ]
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval: float = 300.0  # 5 minutes
        self._callbacks: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize cache manager"""
        # Initialize backends
        self._backends[CacheLevel.L1] = MemoryCacheBackend()
        self._backends[CacheLevel.L2] = DiskCacheBackend()
        self._backends[CacheLevel.L3] = DatabaseCacheBackend()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Refactored cache manager initialized")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired entries"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _cleanup_expired_entries(self) -> None:
        """Cleanup expired entries from all backends"""
        for backend in self._backends.values():
            if hasattr(backend, '_cleanup_expired'):
                await backend._cleanup_expired()
    
    async def get(self, key: str, level: CacheLevel = None) -> Optional[Any]:
        """Get value from cache with fallback"""
        if level:
            # Get from specific level
            backend = self._backends.get(level)
            if backend:
                return await backend.get(key)
            return None
        
        # Try all levels in order
        for cache_level in self._cache_order:
            backend = self._backends.get(cache_level)
            if backend:
                value = await backend.get(key)
                if value is not None:
                    # Promote to higher levels
                    await self._promote_to_higher_levels(key, value, cache_level)
                    return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None,
                 level: CacheLevel = None, strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH) -> bool:
        """Set value in cache"""
        if level:
            # Set in specific level
            backend = self._backends.get(level)
            if backend:
                return await backend.set(key, value, ttl)
            return False
        
        # Set in all levels based on strategy
        if strategy == CacheStrategy.WRITE_THROUGH:
            # Write to all levels
            success = True
            for cache_level in self._cache_order:
                backend = self._backends.get(cache_level)
                if backend:
                    if not await backend.set(key, value, ttl):
                        success = False
            return success
        
        elif strategy == CacheStrategy.WRITE_BACK:
            # Write to highest level only
            backend = self._backends.get(self._cache_order[0])
            if backend:
                return await backend.set(key, value, ttl)
            return False
        
        elif strategy == CacheStrategy.WRITE_AROUND:
            # Write to lower levels only
            success = True
            for cache_level in self._cache_order[1:]:
                backend = self._backends.get(cache_level)
                if backend:
                    if not await backend.set(key, value, ttl):
                        success = False
            return success
        
        return False
    
    async def delete(self, key: str, level: CacheLevel = None) -> bool:
        """Delete value from cache"""
        if level:
            # Delete from specific level
            backend = self._backends.get(level)
            if backend:
                return await backend.delete(key)
            return False
        
        # Delete from all levels
        success = True
        for cache_level in self._cache_order:
            backend = self._backends.get(cache_level)
            if backend:
                if not await backend.delete(key):
                    success = False
        
        return success
    
    async def clear(self, level: CacheLevel = None) -> bool:
        """Clear cache"""
        if level:
            # Clear specific level
            backend = self._backends.get(level)
            if backend:
                return await backend.clear()
            return False
        
        # Clear all levels
        success = True
        for cache_level in self._cache_order:
            backend = self._backends.get(cache_level)
            if backend:
                if not await backend.clear():
                    success = False
        
        return success
    
    async def exists(self, key: str, level: CacheLevel = None) -> bool:
        """Check if key exists in cache"""
        if level:
            # Check specific level
            backend = self._backends.get(level)
            if backend:
                return await backend.exists(key)
            return False
        
        # Check all levels
        for cache_level in self._cache_order:
            backend = self._backends.get(cache_level)
            if backend:
                if await backend.exists(key):
                    return True
        
        return False
    
    async def _promote_to_higher_levels(self, key: str, value: Any, from_level: CacheLevel) -> None:
        """Promote value to higher cache levels"""
        current_index = self._cache_order.index(from_level)
        
        for i in range(current_index):
            higher_level = self._cache_order[i]
            backend = self._backends.get(higher_level)
            if backend:
                await backend.set(key, value)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        
        for level, backend in self._backends.items():
            if hasattr(backend, 'get_stats'):
                stats[level.value] = backend.get_stats()
            else:
                stats[level.value] = {
                    "size": await backend.size(),
                    "hits": 0,
                    "misses": 0,
                    "hit_rate": 0.0
                }
        
        return stats
    
    def add_callback(self, callback: Callable) -> None:
        """Add cache callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove cache callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache manager health status"""
        return {
            "backends_count": len(self._backends),
            "cache_order": [level.value for level in self._cache_order],
            "cleanup_interval": self._cleanup_interval,
            "stats": await self.get_stats()
        }
    
    async def shutdown(self) -> None:
        """Shutdown cache manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Refactored cache manager shutdown")


# Global cache manager
cache_manager = RefactoredCacheManager()


# Convenience functions
async def get_cache(key: str, **kwargs):
    """Get value from cache"""
    return await cache_manager.get(key, **kwargs)


async def set_cache(key: str, value: Any, **kwargs):
    """Set value in cache"""
    return await cache_manager.set(key, value, **kwargs)


async def delete_cache(key: str, **kwargs):
    """Delete value from cache"""
    return await cache_manager.delete(key, **kwargs)


async def clear_cache(**kwargs):
    """Clear cache"""
    return await cache_manager.clear(**kwargs)


# Cache decorators
def cached(ttl: timedelta = None, level: CacheLevel = None, strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH):
    """Cache decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_value = await get_cache(cache_key, level=level)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await set_cache(cache_key, result, ttl=ttl, level=level, strategy=strategy)
            
            return result
        return wrapper
    return decorator


def cache_invalidate(pattern: str = None):
    """Cache invalidation decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            if pattern:
                # This would implement pattern-based invalidation
                pass
            else:
                # Invalidate specific keys
                pass
            
            return result
        return wrapper
    return decorator





















