"""
Retry Handler for Social Video Transcriber AI
Implements retry logic with exponential backoff and circuit breaker pattern
"""

import asyncio
import functools
import logging
import time
from typing import Optional, Callable, Any, Type, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to delays
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0  # seconds before trying half-open
    
    
@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)


class RetryExhausted(Exception):
    """Raised when all retries are exhausted"""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, service_name: str, retry_after: float):
        super().__init__(f"Circuit breaker open for {service_name}")
        self.service_name = service_name
        self.retry_after = retry_after


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for current attempt"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhausted: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_retries:
                    logger.error(f"All {self.config.max_retries} retries exhausted")
                    raise RetryExhausted(
                        f"Failed after {self.config.max_retries} retries",
                        last_exception
                    )
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        raise RetryExhausted("Unexpected retry exit", last_exception)


class CircuitBreaker:
    """Implements circuit breaker pattern"""
    
    def __init__(
        self,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout"""
        if self._state.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._state.last_failure_time:
                elapsed = (datetime.utcnow() - self._state.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
        
        return self._state.state
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.utcnow()
        
        if new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._state.success_count = 0
        
        logger.info(
            f"Circuit breaker [{self.service_name}]: "
            f"{old_state.value} -> {new_state.value}"
        )
    
    def record_success(self):
        """Record a successful call"""
        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            
            if self._state.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        
        elif self._state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self._state.failure_count = 0
    
    def record_failure(self):
        """Record a failed call"""
        self._state.failure_count += 1
        self._state.last_failure_time = datetime.utcnow()
        
        if self._state.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)
        
        elif self._state.state == CircuitState.CLOSED:
            if self._state.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        state = self.state  # This checks for timeout
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return True  # Allow test request
        else:  # OPEN
            return False
    
    def get_retry_after(self) -> float:
        """Get seconds until circuit might close"""
        if self._state.state != CircuitState.OPEN:
            return 0
        
        if self._state.last_failure_time:
            elapsed = (datetime.utcnow() - self._state.last_failure_time).total_seconds()
            return max(0, self.config.timeout - elapsed)
        
        return self.config.timeout
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.allow_request():
            raise CircuitOpenError(
                self.service_name,
                self.get_retry_after()
            )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure()
            raise
    
    def get_status(self) -> dict:
        """Get circuit breaker status"""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "last_failure_time": (
                self._state.last_failure_time.isoformat()
                if self._state.last_failure_time else None
            ),
            "retry_after": self.get_retry_after(),
        }


class ResilientExecutor:
    """Combines retry handler and circuit breaker"""
    
    def __init__(
        self,
        service_name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.retry_handler = RetryHandler(retry_config)
        self.circuit_breaker = CircuitBreaker(service_name, circuit_config)
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with both retry and circuit breaker
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        async def wrapped():
            return await self.circuit_breaker.execute(func, *args, **kwargs)
        
        return await self.retry_handler.execute(wrapped)


# Decorator versions
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for adding retry logic to a function"""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=retryable_exceptions,
    )
    handler = RetryHandler(config)
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await handler.execute(func, *args, **kwargs)
        return wrapper
    
    return decorator


# Global circuit breakers for different services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service"""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(service_name)
    return _circuit_breakers[service_name]












