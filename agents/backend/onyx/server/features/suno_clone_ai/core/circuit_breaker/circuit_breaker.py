"""
Circuit Breaker

Circuit breaker pattern for fault tolerance.
"""

import logging
import time
from enum import Enum
from typing import Callable, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()
    
    def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        with self.lock:
            # Check state
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
                else:
                    # Try to recover
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
            
            # Call function
            try:
                result = func(*args, **kwargs)
                
                # Success - reset on closed or half-open
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED - recovery successful")
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0
                
                return result
            
            except self.expected_exception as e:
                # Failure
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                
                raise
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker reset")
    
    def get_state(self) -> CircuitState:
        """Get current state."""
        return self.state


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    **kwargs
) -> CircuitBreaker:
    """Create circuit breaker."""
    return CircuitBreaker(failure_threshold, recovery_timeout, **kwargs)


def circuit_breaker_decorator(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Failure threshold
        recovery_timeout: Recovery timeout
        
    Returns:
        Decorator function
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator



