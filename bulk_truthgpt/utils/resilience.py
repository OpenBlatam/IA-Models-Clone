"""
Resilience utilities: circuit breakers, retries, and fault tolerance.
"""
import logging
from typing import Callable, Any, Optional
from functools import wraps
from circuitbreaker import circuit
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

logger = logging.getLogger(__name__)


def resilient_call(
    func: Callable,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    circuit_failure_threshold: int = 5,
    circuit_recovery_timeout: int = 60,
):
    """
    Decorator that adds retry logic and circuit breaker to async functions.
    
    Args:
        func: Async function to wrap
        max_retries: Maximum number of retry attempts
        backoff_base: Base time for exponential backoff (seconds)
        circuit_failure_threshold: Number of failures before opening circuit
        circuit_recovery_timeout: Seconds before attempting recovery
    """
    @circuit(
        failure_threshold=circuit_failure_threshold,
        recovery_timeout=circuit_recovery_timeout,
        expected_exception=Exception,
    )
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff_base),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Call to {func.__name__} failed: {str(e)}",
                extra={"function": func.__name__, "error": str(e)},
            )
            raise
    
    return wrapper


class CircuitBreakerState:
    """Track circuit breaker state for observability."""
    def __init__(self):
        self.open_count = 0
        self.half_open_count = 0
        self.closed_count = 0
    
    def record_state_change(self, state: str):
        if state == "open":
            self.open_count += 1
        elif state == "half_open":
            self.half_open_count += 1
        else:
            self.closed_count += 1


# Global circuit breaker state tracker
circuit_breaker_state = CircuitBreakerState()


