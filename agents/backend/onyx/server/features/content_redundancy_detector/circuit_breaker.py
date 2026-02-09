"""
Circuit Breaker Implementation for Resilient Service Communication
Following microservices patterns for fault tolerance and graceful degradation
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes (half-open)
    timeout: int = 60  # Time before attempting recovery (seconds)
    expected_exception: type = Exception  # Exception type that counts as failure
    name: str = "default"


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    failures: int = 0
    successes: int = 0
    total_requests: int = 0
    state_transitions: list = field(default_factory=list)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for resilient microservices
    
    Prevents cascading failures by detecting failures and preventing
    requests when a service is down.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._opened_at: Optional[float] = None
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async or sync function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: When function execution fails
        """
        async with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit is open and timeout has elapsed
            if self.state == CircuitState.OPEN:
                if self._opened_at and time.time() - self._opened_at >= self.config.timeout:
                    logger.info(f"Circuit breaker {self.config.name}: Moving to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.stats.successes = 0
                    self.stats.failures = 0
                    self.stats.state_transitions.append({
                        "from": "open",
                        "to": "half_open",
                        "timestamp": time.time()
                    })
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.config.name} is OPEN"
                    )
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            return result
            
        except self.config.expected_exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self.stats.successes += 1
            self.stats.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.successes >= self.config.success_threshold:
                    logger.info(f"Circuit breaker {self.config.name}: Moving to CLOSED")
                    self.state = CircuitState.CLOSED
                    self._opened_at = None
                    self.stats.failures = 0
                    self.stats.successes = 0
                    self.stats.state_transitions.append({
                        "from": "half_open",
                        "to": "closed",
                        "timestamp": time.time()
                    })
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failures = 0
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.stats.failures += 1
            self.stats.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker {self.config.name}: Moving to OPEN (failure in half-open)")
                self.state = CircuitState.OPEN
                self._opened_at = time.time()
                self.stats.state_transitions.append({
                    "from": "half_open",
                    "to": "open",
                    "timestamp": time.time()
                })
            elif self.state == CircuitState.CLOSED:
                if self.stats.failures >= self.config.failure_threshold:
                    logger.warning(f"Circuit breaker {self.config.name}: Moving to OPEN (threshold exceeded)")
                    self.state = CircuitState.OPEN
                    self._opened_at = time.time()
                    self.stats.state_transitions.append({
                        "from": "closed",
                        "to": "open",
                        "timestamp": time.time()
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "total_requests": self.stats.total_requests,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
            "opened_at": self._opened_at,
            "state_transitions": self.stats.state_transitions
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.stats.failures = 0
            self.stats.successes = 0
            self._opened_at = None
            logger.info(f"Circuit breaker {self.config.name}: Reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker"""
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=name)
        _circuit_breakers[name] = CircuitBreaker(config)
    return _circuit_breakers[name]


def reset_all_circuit_breakers():
    """Reset all circuit breakers (useful for testing)"""
    for breaker in _circuit_breakers.values():
        breaker.reset()






