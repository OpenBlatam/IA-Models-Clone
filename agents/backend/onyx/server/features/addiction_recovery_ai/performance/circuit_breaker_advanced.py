"""
Advanced Circuit Breaker
Circuit breaker with auto-recovery and adaptive thresholds
"""

import logging
import time
import asyncio
from typing import Callable, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Time before trying half-open
    window_size: int = 100  # Window for failure counting
    failure_rate_threshold: float = 0.5  # Failure rate to open (0-1)


class AdvancedCircuitBreaker:
    """
    Advanced circuit breaker
    
    Features:
    - Auto-recovery
    - Adaptive thresholds
    - Failure rate monitoring
    - Time-based recovery
    - Success rate tracking
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        
        self._failure_history: deque = deque(maxlen=self.config.window_size)
        self._success_history: deque = deque(maxlen=self.config.window_size)
        self._state_changes: deque = deque(maxlen=100)
        
        self._stats = {
            "total_requests": 0,
            "total_failures": 0,
            "total_successes": 0,
            "rejected_requests": 0,
            "state_changes": 0
        }
        
        logger.info(f"✅ Advanced circuit breaker '{name}' initialized")
    
    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
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
        """
        self._stats["total_requests"] += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) >= self.config.timeout:
                # Try half-open
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self._record_state_change("OPEN -> HALF_OPEN")
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                # Reject request
                self._stats["rejected_requests"] += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful execution"""
        current_time = time.time()
        self.last_success_time = current_time
        self.success_count += 1
        self._success_history.append(current_time)
        self._stats["total_successes"] += 1
        
        # Update state based on success
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                # Close circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self._record_state_change("HALF_OPEN -> CLOSED")
                logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
        
        # Reset failure count if in closed state and doing well
        if self.state == CircuitState.CLOSED:
            failure_rate = self._calculate_failure_rate()
            if failure_rate < 0.1:  # Less than 10% failures
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self):
        """Record failed execution"""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_count += 1
        self._failure_history.append(current_time)
        self._stats["total_failures"] += 1
        
        # Check if should open circuit
        failure_rate = self._calculate_failure_rate()
        
        if self.state == CircuitState.CLOSED:
            if (self.failure_count >= self.config.failure_threshold or
                failure_rate >= self.config.failure_rate_threshold):
                # Open circuit
                self.state = CircuitState.OPEN
                self._record_state_change("CLOSED -> OPEN")
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED "
                    f"(failures: {self.failure_count}, rate: {failure_rate:.2%})"
                )
        
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self.state = CircuitState.OPEN
            self.success_count = 0
            self._record_state_change("HALF_OPEN -> OPEN")
            logger.warning(f"Circuit breaker '{self.name}' re-OPENED from HALF_OPEN")
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        total = len(self._failure_history) + len(self._success_history)
        if total == 0:
            return 0.0
        
        return len(self._failure_history) / total
    
    def _record_state_change(self, change: str):
        """Record state change"""
        self._state_changes.append({
            "timestamp": time.time(),
            "change": change,
            "state": self.state.value
        })
        self._stats["state_changes"] += 1
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        failure_rate = self._calculate_failure_rate()
        success_rate = 1.0 - failure_rate
        
        return {
            **self._stats,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": failure_rate,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self._failure_history.clear()
        self._success_history.clear()
        logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breakers registry
_circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}


def create_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> AdvancedCircuitBreaker:
    """Create and register circuit breaker"""
    breaker = AdvancedCircuitBreaker(name, config)
    _circuit_breakers[name] = breaker
    return breaker


def get_circuit_breaker(name: str) -> Optional[AdvancedCircuitBreaker]:
    """Get circuit breaker by name"""
    return _circuit_breakers.get(name)















