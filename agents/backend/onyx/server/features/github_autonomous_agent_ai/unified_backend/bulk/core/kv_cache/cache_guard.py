"""
Cache guard for protection and safety.

Provides safety mechanisms and circuit breakers.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, blocking operations
    HALF_OPEN = "half_open"  # Testing if service recovered


class CacheGuard:
    """
    Cache guard with circuit breaker pattern.
    
    Protects cache from overload and failures.
    """
    
    def __init__(
        self,
        cache: Any,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize cache guard.
        
        Args:
            cache: Cache instance
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying half-open
            half_open_max_calls: Max calls in half-open state
        """
        self.cache = cache
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.success_count = 0
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should open."""
        return self.failure_count >= self.failure_threshold
    
    def _should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self._should_open_circuit():
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened due to failures")
    
    def _record_success(self) -> None:
        """Record a success."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker closed after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def call(self, operation: str, *args, **kwargs) -> Any:
        """
        Execute operation with circuit breaker protection.
        
        Args:
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            RuntimeError: If circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        # Execute operation
        try:
            func = getattr(self.cache, operation)
            result = func(*args, **kwargs)
            self._record_success()
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
            
            return result
            
        except Exception as e:
            self._record_failure()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker re-opened after failure in half-open")
            
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get circuit breaker state.
        
        Returns:
            Dictionary with state information
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
            "success_count": self.success_count
        }
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.success_count = 0
        logger.info("Circuit breaker reset")


class CacheRateLimiter:
    """
    Rate limiter for cache operations.
    
    Prevents cache overload.
    """
    
    def __init__(
        self,
        max_operations_per_second: float = 1000.0,
        burst_size: int = 100
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_operations_per_second: Maximum operations per second
            burst_size: Maximum burst size
        """
        self.max_ops_per_sec = max_operations_per_second
        self.burst_size = burst_size
        
        self.operation_times: list[float] = []
        self.last_check = time.time()
    
    def allow(self) -> bool:
        """
        Check if operation is allowed.
        
        Returns:
            True if operation is allowed
        """
        now = time.time()
        
        # Remove old operations
        self.operation_times = [
            t for t in self.operation_times
            if now - t < 1.0
        ]
        
        # Check rate limit
        if len(self.operation_times) >= self.max_ops_per_sec:
            return False
        
        # Check burst limit
        recent_ops = [
            t for t in self.operation_times
            if now - t < 0.1  # 100ms window
        ]
        
        if len(recent_ops) >= self.burst_size:
            return False
        
        # Allow operation
        self.operation_times.append(now)
        return True
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit is reached."""
        if not self.allow():
            # Calculate wait time
            oldest = min(self.operation_times)
            wait_time = 1.0 - (time.time() - oldest)
            if wait_time > 0:
                time.sleep(wait_time)

