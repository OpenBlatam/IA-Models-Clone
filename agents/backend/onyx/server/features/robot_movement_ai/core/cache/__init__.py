"""
Intelligent Cache
=================

Sistema de cache inteligente con TTL y estrategias de eviction.
"""

from .intelligent_cache import (
    IntelligentCache,
    CacheEntry,
    CacheStats,
    TTL,
    LRU,
    LFU,
    FIFO
)

__all__ = [
    "IntelligentCache",
    "CacheEntry",
    "CacheStats",
    "TTL",
    "LRU",
    "LFU",
    "FIFO"
]

