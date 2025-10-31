from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
from typing import Callable, Any
from ...core.interfaces.circuit_breaker_interface import ICircuitBreaker, CircuitState
from ...core.exceptions.api_exceptions import CircuitBreakerOpenException
import logging
from typing import Any, List, Dict, Optional
"""
Circuit Breaker Implementation
==============================

Concrete implementation of circuit breaker pattern.
"""


logger = logging.getLogger(__name__)


class CircuitBreakerService(ICircuitBreaker):
    """Enterprise circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
                 half_open_max_calls: int = 5):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.total_calls = 0
        self.successful_calls = 0
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.total_calls += 1
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioned to HALF_OPEN")
            else:
                raise CircuitBreakerOpenException(
                    service="default",
                    message="Service temporarily unavailable (Circuit Breaker OPEN)"
                )
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenException(
                    service="default",
                    message="Service temporarily unavailable (Circuit Breaker HALF_OPEN max calls exceeded)"
                )
            self.half_open_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> Any:
        """Handle successful call."""
        self.successful_calls += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker transitioned to CLOSED after successful call")
        
        self.failure_count = 0
        self.half_open_calls = 0
    
    async def _on_failure(self) -> Any:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker transitioned to OPEN after {self.failure_count} failures")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
    
    async def reset(self) -> Any:
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        success_rate = (self.successful_calls / self.total_calls) if self.total_calls > 0 else 0.0
        
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "success_rate": success_rate,
            "half_open_calls": self.half_open_calls,
            "half_open_max_calls": self.half_open_max_calls,
            "last_failure_time": self.last_failure_time
        } 