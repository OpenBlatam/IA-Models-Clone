"""
Circuit breaker implementation for resilient service communication
"""
import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them to recover.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: tuple = (Exception,),
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting to close circuit
            expected_exception: Exceptions that count as failures
            name: Name of the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
        self.half_open_success_threshold = 2
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if time.time() - (self.last_failure_time or 0) >= self.timeout:
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Retry after {self.timeout - (time.time() - (self.last_failure_time or 0)):.0f} seconds"
                )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name} transitioning back to OPEN")
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker {self.name} transitioning to OPEN")
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance
    
    Args:
        name: Name of the circuit breaker
        **kwargs: Circuit breaker configuration
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def circuit_breaker(name: str, **breaker_kwargs):
    """
    Decorator for applying circuit breaker to a function
    
    Usage:
        @circuit_breaker("openai_api", failure_threshold=5, timeout=60)
        def call_openai():
            ...
    """
    def decorator(func: Callable):
        breaker = get_circuit_breaker(name, **breaker_kwargs)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator




