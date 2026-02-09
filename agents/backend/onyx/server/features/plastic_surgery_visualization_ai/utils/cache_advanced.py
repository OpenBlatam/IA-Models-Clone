"""Advanced caching utilities."""

from typing import Any, Optional, Callable
from functools import wraps
import time
import hashlib
import json
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timedelta
import aiofiles

from utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Least Recently Used cache."""
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum cache size
        """
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remove oldest (first item)
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class TTLCache:
    """Time To Live cache."""
    
    def __init__(self, ttl_seconds: float = 3600.0):
        """
        Initialize TTL cache.
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/not found
        """
        if key not in self.cache:
            return None
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            self.delete(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL
        """
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        deleted = False
        if key in self.cache:
            del self.cache[key]
            deleted = True
        if key in self.timestamps:
            del self.timestamps[key]
        return deleted
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if now - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)


def cached_lru(maxsize: int = 128):
    """
    Decorator for LRU caching.
    
    Args:
        maxsize: Maximum cache size
    """
    cache = LRUCache(maxsize)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = str((args, tuple(sorted(kwargs.items()))))
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator


def cached_ttl(ttl_seconds: float = 3600.0):
    """
    Decorator for TTL caching.
    
    Args:
        ttl_seconds: Time to live in seconds
    """
    cache = TTLCache(ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = str((args, tuple(sorted(kwargs.items()))))
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator


# File-based cache from cache.py
class FileCache:
    """Simple file-based cache."""
    
    def __init__(self, cache_dir: str = "./storage/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.json"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
                content = await f.read()
                if not content:
                    return None
                data = json.loads(content)
            
            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.utcnow() - cached_time > self.ttl:
                await self.delete(key)
                return None
            
            return data["value"]
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid cache file format for {key}: {e}")
            await self.delete(key)
            return None
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "value": value
            }
            
            async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {key}: {e}")
    
    async def clear(self) -> None:
        """Clear all cache."""
        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {deleted} cache files")
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        
        return {
            "total_entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl.total_seconds() / 3600
        }


# Alias for backward compatibility
Cache = FileCache
