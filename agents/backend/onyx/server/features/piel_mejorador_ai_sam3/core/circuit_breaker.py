"""
Circuit Breaker for Piel Mejorador AI SAM3
==========================================

Circuit breaker pattern for resilient API calls.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Open after N failures
    success_threshold: int = 2  # Close after N successes in half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    expected_exception: type = Exception


class CircuitBreaker:
    """
    Circuit breaker for resilient API calls.
    
    Features:
    - Automatic failure detection
    - Half-open state for recovery testing
    - Configurable thresholds
    - Timeout-based recovery
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
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
        
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opens": 0,
            "circuit_closes": 0,
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        async with self._lock:
            self._stats["total_calls"] += 1
            
            # Check state
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        logger.info(f"Circuit breaker {self.name} entering half-open state")
                        self.state = CircuitState.HALF_OPEN
                        self.success_count = 0
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker {self.name} is OPEN. "
                            f"Wait {self.config.timeout_seconds - elapsed:.1f}s"
                        )
            
            # Execute function
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Success
                await self._on_success()
                self._stats["successful_calls"] += 1
                return result
                
            except self.config.expected_exception as e:
                # Failure
                await self._on_failure()
                self._stats["failed_calls"] += 1
                raise
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker {self.name} closing (recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self._stats["circuit_closes"] += 1
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed in half-open, go back to open
            logger.warning(f"Circuit breaker {self.name} reopening (test failed)")
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.error(f"Circuit breaker {self.name} opening (too many failures)")
                self.state = CircuitState.OPEN
                self._stats["circuit_opens"] += 1
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            **self._stats,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass




