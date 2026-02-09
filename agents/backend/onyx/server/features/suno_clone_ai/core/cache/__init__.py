"""
Advanced Cache Module

Provides:
- Advanced caching strategies
- Cache backends
- Cache utilities
"""

from .cache_backend import (
    CacheBackend,
    MemoryCache,
    FileCache,
    create_cache
)

from .cache_strategies import (
    CacheStrategy,
    LRUCache,
    FIFOCache,
    TTLCache
)

__all__ = [
    # Cache backends
    "CacheBackend",
    "MemoryCache",
    "FileCache",
    "create_cache",
    # Cache strategies
    "CacheStrategy",
    "LRUCache",
    "FIFOCache",
    "TTLCache"
]



