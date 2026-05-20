"""
Advanced Caching Strategies
"""

import torch
import hashlib
import pickle
import time
from typing import Dict, List, Optional, Any, Callable
from collections import OrderedDict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU Cache with TTL and size limits"""
    
    def __init__(
        self,
        maxsize: int = 1000,
        ttl: Optional[float] = None
    ):
        """
        Initialize LRU cache
        
        Args:
            maxsize: Maximum cache size
            ttl: Time to live in seconds (None = no expiration)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if self.ttl is None:
            return False
        
        if key not in self.timestamps:
            return True
        
        return (time.time() - self.timestamps[key]) > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key not in self.cache:
            return None
        
        if self._is_expired(key):
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Check size
            if len(self.cache) >= self.maxsize:
                # Remove oldest
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hit_rate": 0.0,  # Would need to track hits/misses
            "ttl": self.ttl
        }


class PersistentCache:
    """Persistent cache with disk storage"""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        maxsize: int = 10000
    ):
        """
        Initialize persistent cache
        
        Args:
            cache_dir: Cache directory
            maxsize: Maximum cache size
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.maxsize = maxsize
        self.memory_cache = LRUCache(maxsize=maxsize)
        
        logger.info(f"PersistentCache initialized: {cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Store in memory cache
                self.memory_cache.set(key, value)
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        # Store in memory
        self.memory_cache.set(key, value)
        
        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear(self):
        """Clear cache"""
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file: {e}")


class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(
        self,
        cache: LRUCache,
        key_func: Optional[Callable] = None
    ):
        """
        Initialize cache decorator
        
        Args:
            cache: Cache instance
            key_func: Optional function to generate cache key
        """
        self.cache = cache
        self.key_func = key_func or self._default_key
    
    def _default_key(self, *args, **kwargs) -> str:
        """Default key generation"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "|".join(key_parts)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator call"""
        def wrapper(*args, **kwargs):
            key = self.key_func(*args, **kwargs)
            
            # Try cache
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            
            # Compute
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache.set(key, result)
            
            return result
        
        return wrapper

