"""
Cache System Module

High-performance caching system with multiple backends
and intelligent cache management for the AI History Comparison System.
"""

from .cache_manager import CacheManager, CacheConfig
from .cache_backends import (
    MemoryCache, RedisCache, DiskCache, 
    HybridCache, CacheBackend, CacheEntry
)
from .cache_strategies import (
    LRUStrategy, LFUStrategy, TTLStrategy, 
    SizeBasedStrategy, CacheStrategy
)
from .cache_decorators import cached, cached_async, cache_invalidate
from .cache_metrics import CacheMetrics, CacheStats

__all__ = [
    'CacheManager', 'CacheConfig',
    'MemoryCache', 'RedisCache', 'DiskCache', 'HybridCache',
    'CacheBackend', 'CacheEntry',
    'LRUStrategy', 'LFUStrategy', 'TTLStrategy', 'SizeBasedStrategy',
    'CacheStrategy',
    'cached', 'cached_async', 'cache_invalidate',
    'CacheMetrics', 'CacheStats'
]





















