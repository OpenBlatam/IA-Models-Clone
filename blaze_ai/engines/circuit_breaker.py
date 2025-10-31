"""
Circuit Breaker Pattern Implementation for Blaze AI System.

This module provides a robust circuit breaker implementation with
advanced features like half-open state, success thresholds, and
adaptive recovery mechanisms.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Awaitable, Optional, Dict
from contextlib import asynccontextmanager

from ..utils.logging import get_logger

# =============================================================================
# Circuit Breaker States and Configuration
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout: float = 60.0       # Time to wait before half-open
    expected_exception: type = Exception # Exception type to catch
    success_threshold: int = 2           # Successes to close from half-open
    monitoring_window: float = 300.0     # Time window for failure counting
    enable_adaptive_timeout: bool = True # Enable adaptive recovery timeout
    min_timeout: float = 10.0            # Minimum recovery timeout
    max_timeout: float = 300.0           # Maximum recovery timeout
    timeout_multiplier: float = 2.0      # Multiplier for adaptive timeout

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_state_change: float = 0.0
    current_failure_count: int = 0
    current_success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """Enhanced circuit breaker pattern implementation."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.logger = get_logger("circuit_breaker")
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._failure_times: list[float] = []
        self._last_failure_time = 0.0
        self._current_timeout = self.config.recovery_timeout
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit breaker should allow the call
            if not self._should_allow_call():
                self.metrics.rejected_requests += 1
                raise Exception(f"Circuit breaker is {self.state.value} (recovery in {self._get_remaining_timeout():.1f}s)")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except self.config.expected_exception as e:
                await self._on_failure()
                raise e
    
    def _should_allow_call(self) -> bool:
        """Determine if the call should be allowed."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self._last_failure_time >= self._current_timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.logger.info(f"Circuit breaker transitioning to HALF_OPEN state")
    
    def _transition_to_open(self):
        """Transition to open state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self._last_failure_time = time.time()
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.logger.warning(f"Circuit breaker opened after {self.config.failure_threshold} failures")
        
        # Update adaptive timeout if enabled
        if self.config.enable_adaptive_timeout:
            self._update_adaptive_timeout()
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.logger.info(f"Circuit breaker closed - service recovered")
        
        # Reset failure tracking
        self._failure_times.clear()
        self.metrics.current_failure_count = 0
        self.metrics.current_success_count = 0
    
    async def _on_success(self):
        """Handle successful execution."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.metrics.current_success_count += 1
            if self.metrics.current_success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            # Reset failure count on success
            self.metrics.current_failure_count = 0
    
    async def _on_failure(self):
        """Handle failed execution."""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        current_time = time.time()
        
        # Track failure time
        self._failure_times.append(current_time)
        self.metrics.current_failure_count += 1
        
        # Clean old failures outside monitoring window
        self._cleanup_old_failures(current_time)
        
        # Check if circuit should open
        if self.metrics.current_failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _cleanup_old_failures(self, current_time: float):
        """Remove failures outside the monitoring window."""
        cutoff_time = current_time - self.config.monitoring_window
        self._failure_times = [t for t in self._failure_times if t > cutoff_time]
        self.metrics.current_failure_count = len(self._failure_times)
    
    def _update_adaptive_timeout(self):
        """Update timeout using adaptive algorithm."""
        if self.config.enable_adaptive_timeout:
            new_timeout = min(
                self.config.max_timeout,
                max(
                    self.config.min_timeout,
                    self._current_timeout * self.config.timeout_multiplier
                )
            )
            self._current_timeout = new_timeout
            self.logger.info(f"Adaptive timeout updated to {new_timeout:.1f}s")
    
    def _get_remaining_timeout(self) -> float:
        """Get remaining time until circuit can attempt recovery."""
        if self.state != CircuitBreakerState.OPEN:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self._current_timeout - elapsed)
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics
    
    def force_open(self):
        """Force circuit breaker to open state."""
        async def _force_open():
            async with self._lock:
                if self.state != CircuitBreakerState.OPEN:
                    self._transition_to_open()
        
        # Run in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_force_open())
            else:
                loop.run_until_complete(_force_open())
        except RuntimeError:
            # No event loop, just update state
            self.state = CircuitBreakerState.OPEN
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        async def _force_close():
            async with self._lock:
                if self.state != CircuitBreakerState.CLOSED:
                    self._transition_to_closed()
        
        # Run in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_force_close())
            else:
                loop.run_until_complete(_force_close())
        except RuntimeError:
            # No event loop, just update state
            self.state = CircuitBreakerState.CLOSED
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        async def _reset():
            async with self._lock:
                self.state = CircuitBreakerState.CLOSED
                self.metrics = CircuitBreakerMetrics()
                self._failure_times.clear()
                self._current_timeout = self.config.recovery_timeout
                self.logger.info("Circuit breaker reset to initial state")
        
        # Run in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_reset())
            else:
                loop.run_until_complete(_reset())
        except RuntimeError:
            # No event loop, just reset
            self.state = CircuitBreakerState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self._failure_times.clear()
            self._current_timeout = self.config.recovery_timeout

# =============================================================================
# Context Manager for Circuit Breaker
# =============================================================================

@asynccontextmanager
async def circuit_breaker_context(circuit_breaker: CircuitBreaker, func: Callable[..., Awaitable[Any]], *args, **kwargs):
    """Context manager for circuit breaker operations."""
    try:
        result = await circuit_breaker.call(func, *args, **kwargs)
        yield result
    except Exception as e:
        raise e

# =============================================================================
# Factory Functions
# =============================================================================

def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2,
    enable_adaptive_timeout: bool = True
) -> CircuitBreaker:
    """Create a circuit breaker with custom configuration."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        success_threshold=success_threshold,
        enable_adaptive_timeout=enable_adaptive_timeout
    )
    return CircuitBreaker(config)

def create_resilient_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker optimized for resilience."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=1,
        enable_adaptive_timeout=True,
        min_timeout=5.0,
        max_timeout=120.0
    )
    return CircuitBreaker(config)

def create_stable_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker optimized for stability."""
    config = CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=120.0,
        success_threshold=3,
        enable_adaptive_timeout=False
    )
    return CircuitBreaker(config)


