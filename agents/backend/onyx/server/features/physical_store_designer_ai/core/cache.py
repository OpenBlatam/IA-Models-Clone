"""
Centralized caching system for Physical Store Designer AI

This module re-exports all cache components from the refactored cache package
for backward compatibility.
"""

# Import from refactored modules
from .cache import (
    CacheEntry,
    LRUCache,
    CacheManager,
    get_cache_manager,
    cached,
    generate_cache_key,
)

# Re-export for backward compatibility
__all__ = [
    "CacheEntry",
    "LRUCache",
    "CacheManager",
    "get_cache_manager",
    "cached",
    "generate_cache_key",
]

# Note: All cache components are now imported from the refactored cache package.
# The original definitions have been removed.

