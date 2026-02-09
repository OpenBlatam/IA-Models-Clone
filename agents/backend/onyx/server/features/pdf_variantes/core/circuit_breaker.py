"""
PDF Variantes - Circuit Breaker Pattern
Resilient service communication with circuit breakers
"""

import time
import asyncio
from typing import Optional, Callable, Any, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 2           # Successes to close from half-open
    timeout_seconds: int = 60            # Time before trying half-open
    expected_exception: type = Exception  # Exception type that counts as failure


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    failures: int = 0
    successes: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    
    def reset(self):
        """Reset statistics"""
        self.failures = 0
        self.successes = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.last_success_time = None


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Call function with circuit breaker protection"""
        async with self._lock:
            await self._check_state()
        
        # Check if circuit is open
        if self.stats.state == CircuitState.OPEN:
            logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting request")
            if fallback:
                return await self._call_fallback(fallback, *args, **kwargs)
            raise CircuitOpenException(f"Circuit breaker {self.name} is open")
        
        # Try to call function
        try:
            self.stats.total_requests += 1
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success
            await self._record_success()
            return result
        
        except self.config.expected_exception as e:
            # Failure
            await self._record_failure()
            if fallback:
                return await self._call_fallback(fallback, *args, **kwargs)
            raise
    
    async def _check_state(self):
        """Check and update circuit breaker state"""
        now = datetime.utcnow()
        
        if self.stats.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self.stats.last_failure_time:
                elapsed = (now - self.stats.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                    self.stats.state = CircuitState.HALF_OPEN
                    self.stats.successes = 0
        
        elif self.stats.state == CircuitState.HALF_OPEN:
            # Check if we have enough successes
            if self.stats.successes >= self.config.success_threshold:
                logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                self.stats.state = CircuitState.CLOSED
                self.stats.failures = 0
        
        elif self.stats.state == CircuitState.CLOSED:
            # Check if we have too many failures
            if self.stats.failures >= self.config.failure_threshold:
                logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
                self.stats.state = CircuitState.OPEN
                self.stats.last_failure_time = datetime.utcnow()
                self.stats.failures = 0
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.utcnow()
            
            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.successes += 1
            else:
                self.stats.failures = 0
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.utcnow()
            
            if self.stats.state != CircuitState.OPEN:
                self.stats.failures += 1
    
    async def _call_fallback(self, fallback: Callable, *args, **kwargs):
        """Call fallback function"""
        try:
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            return fallback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback function failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
        }
    
    def reset(self):
        """Reset circuit breaker"""
        self.stats.reset()


class CircuitOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]






