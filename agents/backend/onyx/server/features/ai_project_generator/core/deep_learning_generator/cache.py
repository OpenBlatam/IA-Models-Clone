"""
Cache Module

Caching utilities for generator instances and configurations.
"""

from typing import Dict, Any, Optional, Callable
from functools import lru_cache, wraps
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class GeneratorCache:
    """
    Cache for generator instances and configurations.
    """
    
    def __init__(self, max_size: int = 128):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


# Global cache instance
_cache = GeneratorCache()


def cached_generator(max_size: int = 128):
    """
    Decorator to cache generator instances.
    
    Args:
        max_size: Maximum cache size
        
    Example:
        @cached_generator(max_size=64)
        def create_generator(framework, model_type):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache = GeneratorCache(max_size=max_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = cache._make_key(*args, **kwargs)
            
            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {
            "size": cache.size(),
            "max_size": max_size
        }
        
        return wrapper
    return decorator


@lru_cache(maxsize=256)
def cached_config_validation(config_hash: str) -> tuple[bool, Optional[str]]:
    """
    Cached configuration validation.
    
    Args:
        config_hash: Hash of the configuration
        
    Returns:
        Validation result tuple
    """
    # This is a placeholder - actual validation should be done elsewhere
    # This just demonstrates caching pattern
    return True, None


def clear_all_caches() -> None:
    """Clear all caches."""
    _cache.clear()
    cached_config_validation.cache_clear()
    logger.info("All caches cleared")















