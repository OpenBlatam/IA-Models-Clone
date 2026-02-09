"""
Circuit breaker pattern implementation for resilient service calls

This module provides a robust circuit breaker implementation with:
- Configurable thresholds and timeouts
- Metrics collection for observability
- Thread-safe operations
- Sliding window failure tracking
- State transition callbacks
"""

import time
import random
from typing import Callable, Any, Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
import asyncio

from .logging_config import get_logger
from .exceptions import ServiceError

# Import from refactored modules
from .circuit_breaker.circuit_types import CircuitState, CircuitBreakerEventType
from .circuit_breaker.events import CircuitBreakerEvent
from .circuit_breaker.config import CircuitBreakerConfig
from .circuit_breaker.metrics import CircuitBreakerMetrics
from .circuit_breaker.breaker import CircuitBreaker
from .circuit_breaker.registry import (
    circuit_breaker,
    get_circuit_breaker,
    get_circuit_breaker_sync,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
)
from .circuit_breaker.groups import CircuitBreakerGroup
from .circuit_breaker.chain import CircuitBreakerChain
from .circuit_breaker.tracing import get_trace_context, add_tracing_to_circuit_breaker
from .circuit_breaker.store import (
    CircuitBreakerStateStore,
    InMemoryStateStore,
    create_circuit_breaker_with_persistence,
)

logger = get_logger(__name__)


# Re-export for backward compatibility
__all__ = [
    "CircuitState",
    "CircuitBreakerEventType",
    "CircuitBreakerEvent",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreaker",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_circuit_breaker_sync",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    "CircuitBreakerGroup",
    "CircuitBreakerChain",
    "get_trace_context",
    "add_tracing_to_circuit_breaker",
    "CircuitBreakerStateStore",
    "InMemoryStateStore",
    "create_circuit_breaker_with_persistence",
]

# Note: CircuitBreakerConfig, CircuitBreakerMetrics, CircuitBreakerEvent,
# CircuitState, CircuitBreakerEventType, and CircuitBreaker are now imported from
# the refactored modules above. The original definitions have been removed.


# CircuitBreaker class is now imported from .circuit_breaker.breaker

# All components are now imported from refactored modules above.
# The original definitions have been removed.
