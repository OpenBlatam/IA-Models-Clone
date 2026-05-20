"""
Performance Optimization Module
Ultra-fast performance optimizations
"""

from .async_optimizer import (
    AsyncOptimizer,
    async_cache,
    parallelize,
    get_async_optimizer
)
from .serialization_optimizer import (
    SerializationOptimizer,
    FastJSON,
    get_serializer
)
from .database_optimizer import (
    QueryOptimizer,
    ConnectionPoolOptimizer,
    get_query_optimizer,
    get_pool_optimizer
)
from .response_optimizer import (
    FastResponse,
    fast_json_response,
    get_response_optimizer
)
from .memory_optimizer import (
    MemoryOptimizer,
    SlotsMixin,
    get_memory_optimizer
)

__all__ = [
    "AsyncOptimizer",
    "async_cache",
    "parallelize",
    "get_async_optimizer",
    "SerializationOptimizer",
    "FastJSON",
    "get_serializer",
    "QueryOptimizer",
    "ConnectionPoolOptimizer",
    "get_query_optimizer",
    "get_pool_optimizer",
    "FastResponse",
    "fast_json_response",
    "get_response_optimizer",
    "MemoryOptimizer",
    "SlotsMixin",
    "get_memory_optimizer"
]















