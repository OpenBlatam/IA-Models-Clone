"""
Optimizations - Optimizaciones de performance
=============================================

Módulo de optimizaciones para mejorar performance, eficiencia y escalabilidad.
"""

from .performance import (
    optimize_app,
    enable_response_compression,
    enable_connection_pooling,
    enable_query_optimization
)
from .caching import (
    SmartCache,
    CacheDecorator,
    get_smart_cache
)
from .async_optimizations import (
    AsyncBatchProcessor,
    AsyncConnectionPool,
    optimize_async_operations
)
from .memory_optimizations import (
    LazyLoader,
    MemoryOptimizer,
    optimize_memory_usage
)

__all__ = [
    "optimize_app",
    "enable_response_compression",
    "enable_connection_pooling",
    "enable_query_optimization",
    "SmartCache",
    "CacheDecorator",
    "get_smart_cache",
    "AsyncBatchProcessor",
    "AsyncConnectionPool",
    "optimize_async_operations",
    "LazyLoader",
    "MemoryOptimizer",
    "optimize_memory_usage",
]















