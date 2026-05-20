"""
Scalability utilities

This module provides utilities for improving application scalability.
"""

from .connection_pool import ConnectionPool, get_connection_pool
from .background_tasks import BackgroundTaskQueue, background_task_queue
from .circuit_breaker import CircuitBreaker, circuit_breaker_factory

__all__ = [
    "ConnectionPool",
    "get_connection_pool",
    "BackgroundTaskQueue",
    "background_task_queue",
    "CircuitBreaker",
    "circuit_breaker_factory",
]







