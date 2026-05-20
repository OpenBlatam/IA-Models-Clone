"""
Advanced Circuit Breaker System
================================

Advanced circuit breaker with multiple states and strategies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    expected_exception: type = Exception
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with state management."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize advanced circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.half_open_calls = 0
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self.lock:
            # Check state
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure_time:
                    time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                    if time_since_failure >= self.config.timeout_seconds:
                        # Transition to half-open
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                    else:
                        raise Exception(f"Circuit breaker {self.name} is OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            self.total_calls += 1
        
        # Call function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            async with self.lock:
                await self._record_success()
            
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            async with self.lock:
                await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call."""
        self.last_success_time = datetime.now()
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            self.half_open_calls += 1
            
            # Check if we have enough successes
            if self.successes >= self.config.success_threshold:
                # Transition to closed
                self.state = CircuitState.CLOSED
                self.failures = 0
                self.successes = 0
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
            
            # Check if we've exceeded half-open max calls
            if self.half_open_calls >= self.config.half_open_max_calls:
                # Transition back to open
                self.state = CircuitState.OPEN
                self.failures = 0
                self.successes = 0
                self.half_open_calls = 0
                logger.warning(f"Circuit breaker {self.name} transitioning back to OPEN")
        else:
            # Reset failures on success in closed state
            self.failures = 0
    
    async def _record_failure(self):
        """Record failed call."""
        self.last_failure_time = datetime.now()
        self.total_failures += 1
        self.failures += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open transitions to open
            self.state = CircuitState.OPEN
            self.failures = 0
            self.successes = 0
            self.half_open_calls = 0
            logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
        elif self.state == CircuitState.CLOSED:
            # Check if we've exceeded failure threshold
            if self.failures >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
    
    def get_stats(self) -> CircuitBreakerStats:
        """
        Get circuit breaker statistics.
        
        Returns:
            Circuit breaker statistics
        """
        return CircuitBreakerStats(
            state=self.state,
            failures=self.failures,
            successes=self.successes,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            total_calls=self.total_calls,
            total_failures=self.total_failures,
            total_successes=self.total_successes
        )
    
    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self.lock:
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.successes = 0
            self.half_open_calls = 0
            logger.info(f"Circuit breaker {self.name} reset to CLOSED")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
    
    def register(self, name: str, config: CircuitBreakerConfig):
        """
        Register a circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.circuit_breakers[name] = AdvancedCircuitBreaker(name, config)
        logger.info(f"Registered circuit breaker: {name}")
    
    def get(self, name: str) -> Optional[AdvancedCircuitBreaker]:
        """
        Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker or None
        """
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """
        Get statistics for all circuit breakers.
        
        Returns:
            Dictionary of name -> stats
        """
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }



