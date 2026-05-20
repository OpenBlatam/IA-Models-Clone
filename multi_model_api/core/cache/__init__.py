"""
Modular cache system for multi-model API
"""

from .cache import EnhancedCache, get_cache, close_cache
from .stats import CacheStats
from .tags import TagManager
from .decorators import cached, CacheContext
from .layers.l1_cache import CacheEntry
from .constants import (
    DEFAULT_L1_MAX_SIZE,
    DEFAULT_L1_TTL,
    DEFAULT_L2_TTL,
    DEFAULT_COMPRESSION_THRESHOLD,
    MAX_PROMOTION_TASKS,
    REDIS_TIMEOUT,
    REDIS_SCAN_COUNT,
    L3_DEFAULT_SIZE_LIMIT
)

__all__ = [
    "EnhancedCache",
    "CacheStats",
    "CacheEntry",
    "TagManager",
    "cached",
    "CacheContext",
    "get_cache",
    "close_cache",
    "DEFAULT_L1_MAX_SIZE",
    "DEFAULT_L1_TTL",
    "DEFAULT_L2_TTL",
    "DEFAULT_COMPRESSION_THRESHOLD",
    "MAX_PROMOTION_TASKS",
    "REDIS_TIMEOUT",
    "REDIS_SCAN_COUNT",
    "L3_DEFAULT_SIZE_LIMIT"
]

