#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation

Advanced circuit breaker with:
- Fault tolerance and resilience
- Automatic failure detection
- Recovery mechanisms
- Metrics and monitoring
- Configurable thresholds
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import deque

logger = structlog.get_logger("circuit_breaker")

# =============================================================================
# CIRCUIT BREAKER MODELS
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

class CircuitBreakerError(Exception):
    """Circuit breaker specific error."""
    pass

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds to wait before trying again
    success_threshold: int = 3          # Number of successes to close circuit
    timeout: int = 30                   # Request timeout in seconds
    expected_exception: type = Exception  # Exception type to catch
    name: str = "default"               # Circuit breaker name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
            "timeout": self.timeout,
            "expected_exception": self.expected_exception.__name__,
            "name": self.name
        }

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_failure_count: int = 0
    current_success_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "circuit_opened_count": self.circuit_opened_count,
            "circuit_closed_count": self.circuit_closed_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "current_failure_count": self.current_failure_count,
            "current_success_count": self.current_success_count,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "failure_rate": self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        }

# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================

class CircuitBreaker:
    """Advanced circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.failure_times: deque = deque(maxlen=config.failure_threshold)
        self.success_times: deque = deque(maxlen=config.success_threshold)
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time: Optional[datetime] = None
        
        # Response time tracking
        self.response_times: deque = deque(maxlen=100)
        self.slow_request_threshold = 5.0  # seconds
        
        # Monitoring callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self.on_failure: Optional[Callable[[Exception], None]] = None
        self.on_success: Optional[Callable[[Any], None]] = None
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(f"Circuit breaker '{self.config.name}' is OPEN")
        
        # Execute the function
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Add timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            # Call success callback
            if self.on_success:
                self.on_success(result)
            
            return result
            
        except asyncio.TimeoutError:
            # Record timeout as failure
            response_time = time.time() - start_time
            self._record_failure(TimeoutError(f"Request timeout after {self.config.timeout}s"), response_time)
            raise CircuitBreakerError(f"Circuit breaker '{self.config.name}' timeout")
            
        except self.config.expected_exception as e:
            # Record expected exception as failure
            response_time = time.time() - start_time
            self._record_failure(e, response_time)
            raise
            
        except Exception as e:
            # Record unexpected exception as failure
            response_time = time.time() - start_time
            self._record_failure(e, response_time)
            raise CircuitBreakerError(f"Circuit breaker '{self.config.name}' error: {str(e)}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _record_success(self, response_time: float) -> None:
        """Record a successful request."""
        current_time = datetime.utcnow()
        
        self.stats.successful_requests += 1
        self.stats.current_success_count += 1
        self.stats.current_failure_count = 0
        self.stats.last_success_time = current_time
        
        self.success_times.append(current_time)
        self.response_times.append(response_time)
        
        # Check if we should close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.current_success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        # Log slow requests
        if response_time > self.slow_request_threshold:
            logger.warning(
                "Slow request detected",
                circuit_breaker=self.config.name,
                response_time=response_time,
                threshold=self.slow_request_threshold
            )
    
    def _record_failure(self, exception: Exception, response_time: float) -> None:
        """Record a failed request."""
        current_time = datetime.utcnow()
        
        self.stats.failed_requests += 1
        self.stats.current_failure_count += 1
        self.stats.current_success_count = 0
        self.stats.last_failure_time = current_time
        
        self.failure_times.append(current_time)
        self.response_times.append(response_time)
        
        # Call failure callback
        if self.on_failure:
            self.on_failure(exception)
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self.stats.current_failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_change_time = datetime.utcnow()
        self.stats.circuit_opened_count += 1
        
        logger.warning(
            "Circuit breaker opened",
            circuit_breaker=self.config.name,
            failure_count=self.stats.current_failure_count,
            threshold=self.config.failure_threshold
        )
        
        # Call state change callback
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = datetime.utcnow()
        
        logger.info(
            "Circuit breaker half-open",
            circuit_breaker=self.config.name,
            recovery_timeout=self.config.recovery_timeout
        )
        
        # Call state change callback
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.utcnow()
        self.stats.circuit_closed_count += 1
        
        logger.info(
            "Circuit breaker closed",
            circuit_breaker=self.config.name,
            success_count=self.stats.current_success_count,
            threshold=self.config.success_threshold
        )
        
        # Call state change callback
        if self.on_state_change:
            self.on_state_change(old_state, self.state)
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        stats_dict = self.stats.to_dict()
        
        # Add response time statistics
        if self.response_times:
            stats_dict.update({
                "avg_response_time": statistics.mean(self.response_times),
                "min_response_time": min(self.response_times),
                "max_response_time": max(self.response_times),
                "p95_response_time": self._percentile(self.response_times, 95),
                "p99_response_time": self._percentile(self.response_times, 99)
            })
        
        # Add state information
        stats_dict.update({
            "current_state": self.state.value,
            "state_change_time": self.state_change_time.isoformat() if self.state_change_time else None,
            "time_in_current_state": (datetime.utcnow() - self.state_change_time).total_seconds() if self.state_change_time else 0
        })
        
        return stats_dict
    
    def _percentile(self, data: deque, percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.utcnow()
        
        # Reset counters
        self.stats.current_failure_count = 0
        self.stats.current_success_count = 0
        
        # Clear time tracking
        self.failure_times.clear()
        self.success_times.clear()
        
        logger.info(
            "Circuit breaker manually reset",
            circuit_breaker=self.config.name
        )
        
        # Call state change callback
        if self.on_state_change:
            self.on_state_change(old_state, self.state)

# =============================================================================
# CIRCUIT BREAKER MANAGER
# =============================================================================

class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create a new circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(name=name)
        else:
            config.name = name
        
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(
            "Circuit breaker created",
            name=name,
            config=config.to_dict()
        )
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_or_create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        circuit_breaker = self.get_circuit_breaker(name)
        if circuit_breaker is None:
            circuit_breaker = self.create_circuit_breaker(name, config)
        return circuit_breaker
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            logger.info("Circuit breaker removed", name=name)
            return True
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all circuit breakers."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return {}
        
        total_requests = sum(stats['total_requests'] for stats in all_stats.values())
        total_successes = sum(stats['successful_requests'] for stats in all_stats.values())
        total_failures = sum(stats['failed_requests'] for stats in all_stats.values())
        total_opened = sum(stats['circuit_opened_count'] for stats in all_stats.values())
        total_closed = sum(stats['circuit_closed_count'] for stats in all_stats.values())
        
        return {
            'total_circuit_breakers': len(self.circuit_breakers),
            'total_requests': total_requests,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'global_success_rate': total_successes / total_requests if total_requests > 0 else 0,
            'global_failure_rate': total_failures / total_requests if total_requests > 0 else 0,
            'total_circuit_opens': total_opened,
            'total_circuit_closes': total_closed,
            'circuit_breakers': all_stats
        }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        
        logger.info("All circuit breakers reset")

# =============================================================================
# DECORATOR FOR CIRCUIT BREAKER
# =============================================================================

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker functionality to async functions."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        cb = circuit_breaker_manager.get_or_create_circuit_breaker(name, config)
        
        async def wrapper(*args, **kwargs) -> Any:
            return await cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# GLOBAL CIRCUIT BREAKER INSTANCES
# =============================================================================

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

# Pre-configured circuit breakers for common services
video_processor_cb = circuit_breaker_manager.create_circuit_breaker(
    "video_processor",
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        success_threshold=2,
        timeout=60,
        name="video_processor"
    )
)

database_cb = circuit_breaker_manager.create_circuit_breaker(
    "database",
    CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
        success_threshold=3,
        timeout=30,
        name="database"
    )
)

cache_cb = circuit_breaker_manager.create_circuit_breaker(
    "cache",
    CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=30,
        success_threshold=5,
        timeout=10,
        name="cache"
    )
)

external_api_cb = circuit_breaker_manager.create_circuit_breaker(
    "external_api",
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=120,
        success_threshold=2,
        timeout=45,
        name="external_api"
    )
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CircuitState',
    'CircuitBreakerError',
    'CircuitBreakerConfig',
    'CircuitBreakerStats',
    'CircuitBreaker',
    'CircuitBreakerManager',
    'circuit_breaker',
    'circuit_breaker_manager',
    'video_processor_cb',
    'database_cb',
    'cache_cb',
    'external_api_cb'
]





























