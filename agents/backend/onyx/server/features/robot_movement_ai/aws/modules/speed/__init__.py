"""
Speed Optimizations
===================

Ultra-fast performance optimizations.
"""

from aws.modules.speed.cache_warmer import CacheWarmer
from aws.modules.speed.connection_pooler import ConnectionPooler
from aws.modules.speed.compression import CompressionManager, CompressionType
from aws.modules.speed.query_optimizer import QueryOptimizer
from aws.modules.speed.preloader import Preloader, PreloadTask
from aws.modules.speed.response_cache import ResponseCache
from aws.modules.speed.batch_processor import BatchProcessor, BatchConfig

__all__ = [
    "CacheWarmer",
    "ConnectionPooler",
    "CompressionManager",
    "CompressionType",
    "QueryOptimizer",
    "Preloader",
    "PreloadTask",
    "ResponseCache",
    "BatchProcessor",
    "BatchConfig",
]
