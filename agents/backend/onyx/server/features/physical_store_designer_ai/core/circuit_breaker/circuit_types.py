"""
Circuit Breaker Types and Enums

Defines core types, enums, and data structures for circuit breaker.
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Avoid circular imports


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerEventType(Enum):
    """Types of circuit breaker events"""
    STATE_CHANGED = "state_changed"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_HALF_OPENED = "circuit_half_opened"
    FAILURE_RECORDED = "failure_recorded"
    SUCCESS_RECORDED = "success_recorded"
    REQUEST_REJECTED = "request_rejected"
    RETRY_ATTEMPTED = "retry_attempted"
    FALLBACK_USED = "fallback_used"
    TIMEOUT_OCCURRED = "timeout_occurred"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    METRICS_UPDATED = "metrics_updated"

