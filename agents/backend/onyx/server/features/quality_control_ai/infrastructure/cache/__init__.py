"""
Caching Layer

Caching for inspection results and model outputs.
"""

from .cache_manager import CacheManager, get_cache_manager

__all__ = [
    "CacheManager",
    "get_cache_manager",
]



