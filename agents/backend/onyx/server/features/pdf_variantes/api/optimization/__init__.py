"""
Optimization Module
Performance optimization utilities
"""

from .query_cache import QueryCache, cache_query
from .async_pool import AsyncPool, ParallelExecutor, async_parallel
from .db_optimizer import QueryOptimizer, optimize_query, ConnectionPool
from .response_optimizer import ResponseOptimizer, optimize_response, ChunkedResponse
from .lazy_loading import LazyLoader, async_lazy_property, PrefetchOptimizer
from .performance_config import PerformanceConfig

__all__ = [
    "QueryCache",
    "cache_query",
    "AsyncPool",
    "ParallelExecutor",
    "async_parallel",
    "QueryOptimizer",
    "optimize_query",
    "ConnectionPool",
    "ResponseOptimizer",
    "optimize_response",
    "ChunkedResponse",
    "LazyLoader",
    "async_lazy_property",
    "PrefetchOptimizer",
    "PerformanceConfig"
]






