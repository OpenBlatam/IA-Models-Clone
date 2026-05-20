"""
Circuit Breaker Pattern
=======================

Implementation of circuit breaker pattern for resilient service calls.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes (half-open)
    timeout_seconds: float = 60.0  # Time before trying half-open
    expected_exception: type = Exception  # Exception type to catch


class CircuitBreaker:
    """Circuit breaker for resilient service calls."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function call fails
        """
        async with self._lock:
            # Check if circuit should transition
            await self._check_state_transition()
            
            # Reject if open
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        # Try to call function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _check_state_transition(self):
        """Check and perform state transitions."""
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Already handled in _on_success
            pass
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN (failure in half-open)")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN ({self.failure_count} failures)")
    
    def get_state(self) -> CircuitState:
        """Get current state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """Registry for circuit breakers."""
    
    def __init__(self):
        """Initialize registry."""
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
            
        Returns:
            Circuit breaker instance
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }




