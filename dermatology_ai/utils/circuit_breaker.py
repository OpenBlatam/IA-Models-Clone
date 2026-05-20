"""
Circuit Breaker Pattern for resilient service communication
"""

import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout: float = 60.0  # Time to wait before trying half-open
    expected_exception: type = Exception  # Exception type to catch


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
    
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
            CircuitBreakerOpenError: If circuit is open
            Exception: If function call fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, go back to open
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            # Too many failures, open circuit
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        if self.state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.success_count = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info(f"Circuit breaker '{self.name}' closed after recovery")
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]















