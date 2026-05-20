"""
Circuit breaker utilities
Circuit breaker pattern for resilience
"""

from typing import Callable, Optional, TypeVar
from enum import Enum
from datetime import datetime
import asyncio

try:
    from utils.date_helpers import get_current_utc
except ImportError:
    from ..date_helpers import get_current_utc

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable[[], T]) -> T:
        """
        Call function with circuit breaker
        
        Args:
            func: Function to call
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError if circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = get_current_utc()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        if not self.last_failure_time:
            return True
        
        elapsed = (get_current_utc() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED


class CircuitBreakerOpenError(Exception):
    """Error raised when circuit breaker is open"""
    pass


def create_circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0
) -> CircuitBreaker:
    """Create new circuit breaker"""
    return CircuitBreaker(failure_threshold, timeout)

