"""
Cache Backend

Different cache backend implementations.
"""

import logging
import pickle
from typing import Any, Optional, Dict
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Base cache backend."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Get cache file path."""
        # Sanitize key for filename
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        path = self._get_path(key)
        
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        path = self._get_path(key)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        path = self._get_path(key)
        
        if path.exists():
            path.unlink()
    
    def clear(self) -> None:
        """Clear all cache."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def create_cache(
    backend: str = "memory",
    **kwargs
) -> CacheBackend:
    """
    Create cache backend.
    
    Args:
        backend: Backend type ('memory', 'file')
        **kwargs: Backend-specific arguments
        
    Returns:
        CacheBackend instance
    """
    if backend == "memory":
        return MemoryCache(**kwargs)
    elif backend == "file":
        return FileCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache backend: {backend}")

