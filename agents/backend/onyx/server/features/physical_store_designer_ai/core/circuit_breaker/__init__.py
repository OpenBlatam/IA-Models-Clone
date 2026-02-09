"""
Circuit Breaker Module - Refactored Components

This package contains the refactored types, config, metrics, and events.

For the main CircuitBreaker class and functionality, import from the parent module:
    from core.circuit_breaker import CircuitBreaker
"""

# Export the refactored components
from .circuit_types import CircuitState, CircuitBreakerEventType
from .config import CircuitBreakerConfig
from .metrics import CircuitBreakerMetrics
from .events import CircuitBreakerEvent
from .breaker import CircuitBreaker
from .registry import (
    circuit_breaker,
    get_circuit_breaker,
    get_circuit_breaker_sync,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
)
from .groups import CircuitBreakerGroup
from .chain import CircuitBreakerChain
from .tracing import get_trace_context, add_tracing_to_circuit_breaker
from .store import (
    CircuitBreakerStateStore,
    InMemoryStateStore,
    create_circuit_breaker_with_persistence,
)

__all__ = [
    # Types
    "CircuitState",
    "CircuitBreakerEventType",
    # Config
    "CircuitBreakerConfig",
    # Metrics
    "CircuitBreakerMetrics",
    # Events
    "CircuitBreakerEvent",
    # Main class
    "CircuitBreaker",
    # Registry
    "circuit_breaker",
    "get_circuit_breaker",
    "get_circuit_breaker_sync",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    # Groups and Chain
    "CircuitBreakerGroup",
    "CircuitBreakerChain",
    # Tracing
    "get_trace_context",
    "add_tracing_to_circuit_breaker",
    # Persistence
    "CircuitBreakerStateStore",
    "InMemoryStateStore",
    "create_circuit_breaker_with_persistence",
]

