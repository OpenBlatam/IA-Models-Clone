"""
Circuit Breaker Pattern Implementation
Prevents cascading failures in distributed systems
"""

import time
import logging
from enum import Enum
from typing import Callable, Any
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation
    
    Prevents cascading failures by:
    - Opening circuit after failure threshold
    - Rejecting requests when open
    - Testing recovery in half-open state
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state = CircuitState.CLOSED
        self.lock = Lock()
    
    def __enter__(self):
        """Context manager entry"""
        self._check_state()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type and issubclass(exc_type, self.expected_exception):
            self._record_failure()
            return False
        else:
            self._record_success()
            return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        self._check_state()
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise
    
    def _check_state(self) -> None:
        """Check and update circuit breaker state"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if time.time() - self.last_failure_time >= self.timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. "
                        f"Retry after {self.timeout - (time.time() - self.last_failure_time):.1f} seconds"
                    )
    
    def _record_failure(self) -> None:
        """Record a failure"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed in half-open, go back to open
                logger.warning("Circuit breaker transitioning to OPEN (half-open test failed)")
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                # Too many failures, open circuit
                logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
                self.state = CircuitState.OPEN
    
    def _record_success(self) -> None:
        """Record a success"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                # Success in half-open, close circuit
                logger.info("Circuit breaker transitioning to CLOSED (recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0
            logger.info("Circuit breaker manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state















