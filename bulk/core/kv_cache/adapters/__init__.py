"""
Cache adapters module.

Provides adapter classes for different cache implementations.
"""
from kv_cache.base import BaseKVCache
from kv_cache.adapters.adaptive_cache import AdaptiveKVCache
from kv_cache.adapters.paged_cache import PagedKVCache

__all__ = ["BaseKVCache", "AdaptiveKVCache", "PagedKVCache"]

