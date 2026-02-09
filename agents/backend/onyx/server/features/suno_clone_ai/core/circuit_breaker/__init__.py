"""
Circuit Breaker Module

Provides:
- Circuit breaker pattern
- Fault tolerance
- Resilience utilities
"""

from .circuit_breaker import (
    CircuitBreaker,
    create_circuit_breaker,
    circuit_breaker_decorator
)

__all__ = [
    "CircuitBreaker",
    "create_circuit_breaker",
    "circuit_breaker_decorator"
]



