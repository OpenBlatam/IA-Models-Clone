"""
Cache eviction strategies module.

Provides different eviction strategies following best practices.
"""
from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.strategies.lru import LRUEvictionStrategy
from kv_cache.strategies.lfu import LFUEvictionStrategy
from kv_cache.strategies.adaptive import AdaptiveEvictionStrategy
from kv_cache.strategies.factory import create_eviction_strategy

__all__ = [
    "BaseEvictionStrategy",
    "LRUEvictionStrategy",
    "LFUEvictionStrategy",
    "AdaptiveEvictionStrategy",
    "create_eviction_strategy",
]

