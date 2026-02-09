"""
Cache utilities module

This module provides cache-related utilities and helpers.
"""

from ..cache import CacheService, cache_service
from .helpers import (
    get_cached_or_fetch,
    cache_entity,
    invalidate_cache,
    invalidate_cache_pattern,
    invalidate_related_caches,
)

__all__ = [
    "CacheService",
    "cache_service",
    "get_cached_or_fetch",
    "cache_entity",
    "invalidate_cache",
    "invalidate_cache_pattern",
    "invalidate_related_caches",
]


