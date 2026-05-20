"""
Circuit Breaker Service
=======================
Service for implementing circuit breaker pattern
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures to open circuit
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    expected_exception: type = Exception  # Exception type to catch


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState
    failures: int = 0
    successes: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    last_failure: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    last_success: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker for a specific service.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats(state=self.state)
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        func: Callable[[], Awaitable[T]]
    ) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            # Check if circuit should transition
            await self._check_state_transition()
            
            # Reject if circuit is open
            if self.state == CircuitState.OPEN:
                self.stats.rejected_requests += 1
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute function
        self.stats.total_requests += 1
        
        try:
            result = await func()
            await self._record_success()
            return result
        
        except self.config.expected_exception as e:
            await self._record_failure()
            raise
    
    async def _check_state_transition(self):
        """Check and perform state transitions"""
        now = datetime.now()
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.stats.opened_at:
                elapsed = (now - self.stats.opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.stats.state = self.state
                    self.stats.successes = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Already handled in _record_success/_record_failure
            pass
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self.stats.successes += 1
            self.stats.last_success = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.stats.state = self.state
                    self.stats.failures = 0
                    self.stats.opened_at = None
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failures = 0
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.stats.failures += 1
            self.stats.last_failure = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                if self.stats.failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.stats.state = self.state
                    self.stats.opened_at = datetime.now()
                    logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitState.OPEN
                self.stats.state = self.state
                self.stats.opened_at = datetime.now()
                self.stats.successes = 0
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN (from HALF_OPEN)")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failures': self.stats.failures,
            'successes': self.stats.successes,
            'total_requests': self.stats.total_requests,
            'rejected_requests': self.stats.rejected_requests,
            'last_failure': self.stats.last_failure.isoformat() if self.stats.last_failure else None,
            'last_success': self.stats.last_success.isoformat() if self.stats.last_success else None,
            'opened_at': self.stats.opened_at.isoformat() if self.stats.opened_at else None
        }


class CircuitBreakerService:
    """
    Service for managing multiple circuit breakers.
    """
    
    def __init__(self):
        """Initialize circuit breaker service"""
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Optional configuration (uses default if not provided)
        
        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            config = config or CircuitBreakerConfig()
            self._breakers[name] = CircuitBreaker(name, config)
            logger.info(f"Created circuit breaker '{name}'")
        
        return self._breakers[name]
    
    def get_breaker_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get circuit breaker statistics.
        
        Args:
            name: Optional breaker name (returns all if not provided)
        
        Returns:
            Dictionary with statistics
        """
        if name:
            breaker = self._breakers.get(name)
            return breaker.get_stats() if breaker else {}
        
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }


# Global circuit breaker service instance
_circuit_breaker_service: Optional[CircuitBreakerService] = None


def get_circuit_breaker_service() -> CircuitBreakerService:
    """Get or create circuit breaker service instance"""
    global _circuit_breaker_service
    if _circuit_breaker_service is None:
        _circuit_breaker_service = CircuitBreakerService()
    return _circuit_breaker_service

