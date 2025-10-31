"""
Advanced Circuit Breaker Pattern - Fault tolerance and resilience
Production-ready circuit breaker with automatic recovery
"""

import asyncio
import time
import threading
from typing import Any, Dict, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Open after N failures
    success_threshold: int = 2      # Close after N successes
    timeout: float = 60.0           # Time before half-open (seconds)
    timeout_window: float = 300.0  # Time window for failure counting
    expected_exception: type = Exception

class CircuitBreaker:
    """Advanced circuit breaker with automatic recovery"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        
        # Failure history
        self.failure_history: deque = deque(maxlen=100)
        self.success_history: deque = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_timeouts = 0
        self.total_rejected = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_requests += 1
        
        # Check state
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.timeout):
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                self.total_rejected += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {time.time() - (self.last_failure_time or 0):.1f}s ago"
                )
        
        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exception - don't count as failure
            logger.warning(f"Circuit breaker '{self.name}' unexpected exception: {e}")
            raise

    def _record_success(self):
        """Record successful operation"""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                self.last_success_time = current_time
                
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
            elif self.state == CircuitState.OPEN:
                # Shouldn't happen, but handle it
                self.state = CircuitState.HALF_OPEN
                self.success_count = 1
                self.last_success_time = current_time
            else:
                # CLOSED state - reset failure count
                self.failure_count = 0
            
            self.success_history.append(current_time)
            self._cleanup_old_failures(current_time)

    def _record_failure(self, error: Exception):
        """Record failed operation"""
        with self.lock:
            current_time = time.time()
            self.failure_count += 1
            self.last_failure_time = current_time
            self.total_failures += 1
            
            self.failure_history.append({
                "time": current_time,
                "error": str(error),
                "type": type(error).__name__
            })
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed during half-open, go back to open
                self._open_circuit()
            elif self.state == CircuitState.CLOSED:
                # Check if threshold reached
                if self._should_open_circuit():
                    self._open_circuit()
            
            self._cleanup_old_failures(current_time)

    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        if self.failure_count < self.config.failure_threshold:
            return False
        
        # Check failures within timeout window
        current_time = time.time()
        window_start = current_time - self.config.timeout_window
        
        recent_failures = [
            f for f in self.failure_history
            if f["time"] >= window_start
        ]
        
        return len(recent_failures) >= self.config.failure_threshold

    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        self.failure_count = 0  # Reset for next cycle
        self.success_count = 0
        logger.warning(
            f"Circuit breaker '{self.name}' OPENED. "
            f"Will retry after {self.config.timeout}s"
        )

    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")

    def _cleanup_old_failures(self, current_time: float):
        """Remove old failures outside timeout window"""
        window_start = current_time - self.config.timeout_window
        
        while self.failure_history and self.failure_history[0]["time"] < window_start:
            self.failure_history.popleft()

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        with self.lock:
            return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_rejected": self.total_rejected,
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time,
                "recent_failures": len(self.failure_history),
                "recent_successes": len(self.success_history)
            }

    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass
