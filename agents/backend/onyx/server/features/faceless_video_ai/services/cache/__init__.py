"""
Cache system for video generation
Supports both in-memory and persistent caching
"""

from .cache_manager import CacheManager, get_cache_manager

__all__ = ["CacheManager", "get_cache_manager"]

