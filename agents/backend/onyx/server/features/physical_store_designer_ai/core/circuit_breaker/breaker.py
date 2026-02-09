"""
Circuit Breaker Implementation

Main CircuitBreaker class for resilient service calls.
"""

import time
import random
from typing import Callable, Any, Optional, Dict, List, Tuple
from datetime import datetime
import asyncio
import logging

# Import from refactored modules
from .circuit_types import CircuitState, CircuitBreakerEventType
from .events import CircuitBreakerEvent
from .config import CircuitBreakerConfig
from .metrics import CircuitBreakerMetrics

# Import ServiceError - try relative import first, fallback if needed
try:
    from ...exceptions import ServiceError
    from ...logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for direct imports
    logger = logging.getLogger(__name__)
    class ServiceError(Exception):
        def __init__(self, service: str, message: str, details: Optional[Dict] = None):
            self.service = service
            self.message = message
            self.details = details or {}
            super().__init__(f"{service}: {message}")


class CircuitBreaker:
    """Circuit breaker for resilient service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit (deprecated, use config)
            recovery_timeout: Seconds to wait before attempting recovery (deprecated, use config)
            expected_exception: Exception type that triggers circuit breaker (deprecated, use config)
            name: Name for the circuit breaker
            config: Circuit breaker configuration (takes precedence over individual params)
        """
        # Support both old and new API for backward compatibility
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception
            )
        
        self.config = config
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._failure_times: List[float] = []  # Track failure times for sliding window
        self._last_failure_time: Optional[float] = None
        self._current_timeout = config.recovery_timeout
        self._state_callbacks: Dict[CircuitState, List[Callable]] = {
            state: [] for state in CircuitState
        }
        # Half-open rate limiting
        self._half_open_semaphore: Optional[asyncio.Semaphore] = None
        if config.half_open_max_concurrent > 0:
            self._half_open_semaphore = asyncio.Semaphore(config.half_open_max_concurrent)
        # Event handlers
        self._event_handlers: List[Callable[[CircuitBreakerEvent], None]] = []
        self._event_history: List[CircuitBreakerEvent] = []  # Keep last N events
        self._max_event_history: int = 100  # Maximum events to keep in history
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        start_time = time.time()
        
        # Fast path check without lock for better performance
        if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset_fast():
                    # Try fallback if enabled
                    if self.config.fallback_enabled and self.config.fallback_func:
                        self.metrics.fallback_count += 1
                        self._emit_event(
                            CircuitBreakerEventType.FALLBACK_USED,
                            remaining_timeout=self._get_remaining_timeout()
                        )
                        if asyncio.iscoroutinefunction(self.config.fallback_func):
                            return await self.config.fallback_func(*args, **kwargs)
                        else:
                            return self.config.fallback_func(*args, **kwargs)
                    
                    self.metrics.rejected_requests += 1
                    remaining_time = self._get_remaining_timeout()
                    self._emit_event(
                        CircuitBreakerEventType.REQUEST_REJECTED,
                        state=self.state.value,
                        remaining_timeout=remaining_time
                    )
                    raise ServiceError(
                        service=self.name,
                        message=f"Circuit breaker is {self.state.value}. Recovery in {remaining_time:.1f}s",
                        details={
                            "state": self.state.value,
                            "failure_count": self.metrics.current_failure_count,
                            "remaining_timeout": remaining_time
                        }
                    )
        
        # Check if call should be allowed (with lock for state transitions)
        if not await self._should_allow_call():
            # Fallback if enabled
            if self.config.fallback_enabled and self.config.fallback_func:
                self.metrics.fallback_count += 1
                if asyncio.iscoroutinefunction(self.config.fallback_func):
                    return await self.config.fallback_func(*args, **kwargs)
                else:
                    return self.config.fallback_func(*args, **kwargs)
            
            self.metrics.rejected_requests += 1
            remaining_time = self._get_remaining_timeout()
            raise ServiceError(
                service=self.name,
                message=f"Circuit breaker is {self.state.value}. Recovery in {remaining_time:.1f}s",
                details={
                    "state": self.state.value,
                    "failure_count": self.metrics.current_failure_count,
                    "remaining_timeout": remaining_time
                }
            )
        
        # Handle half-open rate limiting
        semaphore = None
        if self.state == CircuitState.HALF_OPEN and self._half_open_semaphore:
            semaphore = self._half_open_semaphore
        
        # Execute with retry if enabled
        if self.config.retry_enabled:
            return await self._call_with_retry(func, semaphore, start_time, *args, **kwargs)
        else:
            return await self._execute_call(func, semaphore, start_time, *args, **kwargs)
    
    def _should_attempt_reset_fast(self) -> bool:
        """Fast check without lock to see if reset should be attempted"""
        if not self._last_failure_time:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self._current_timeout
    
    async def _call_with_retry(self, func: Callable, semaphore: Optional[asyncio.Semaphore], 
                              start_time: float, *args, **kwargs) -> Any:
        """Execute call with retry and exponential backoff"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._execute_call(func, semaphore, start_time, *args, **kwargs)
                if attempt > 0:
                    self.metrics.retry_count += attempt
                return result
            except (self.config.expected_exception, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    # Calculate backoff delay
                    delay = min(
                        self.config.retry_backoff_base * (2 ** attempt),
                        self.config.retry_backoff_max
                    )
                    if self.config.retry_jitter:
                        delay = delay * (0.5 + random.random())
                    await asyncio.sleep(delay)
                    self.metrics.retry_count += 1
                    self._emit_event(
                        CircuitBreakerEventType.RETRY_ATTEMPTED,
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        delay=delay,
                        error_type=type(e).__name__
                    )
                else:
                    # Last attempt failed
                    await self._on_failure()
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    async def _execute_call(self, func: Callable, semaphore: Optional[asyncio.Semaphore],
                           start_time: float, *args, **kwargs) -> Any:
        """Execute the actual function call"""
        # Acquire semaphore if in half-open state
        if semaphore:
            async with semaphore:
                return await self._do_call(func, start_time, *args, **kwargs)
        else:
            return await self._do_call(func, start_time, *args, **kwargs)
    
    async def _do_call(self, func: Callable, start_time: float, *args, **kwargs) -> Any:
        """Perform the actual function execution"""
        try:
            if self.config.call_timeout:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.call_timeout
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args, **kwargs),
                        timeout=self.config.call_timeout
                    )
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            # Record response time and success
            duration = time.time() - start_time
            self.metrics.record_response_time(duration)
            await self._on_success()
            self._emit_event(
                CircuitBreakerEventType.SUCCESS_RECORDED,
                duration=duration,
                response_time=duration
            )
            return result
        
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.metrics.record_response_time(duration)
            await self._on_failure()
            self._emit_event(
                CircuitBreakerEventType.TIMEOUT_OCCURRED,
                timeout=self.config.call_timeout,
                duration=duration
            )
            raise ServiceError(
                service=self.name,
                message=f"Call timeout after {self.config.call_timeout}s",
                details={"timeout": self.config.call_timeout}
            )
        except self.config.expected_exception as e:
            duration = time.time() - start_time
            self.metrics.record_response_time(duration)
            await self._on_failure()
            self._emit_event(
                CircuitBreakerEventType.FAILURE_RECORDED,
                error_type=type(e).__name__,
                error_message=str(e),
                duration=duration,
                failure_count=self.metrics.current_failure_count
            )
            logger.warning(
                f"Circuit breaker {self.name} recorded failure: {e}",
                extra={
                    "state": self.state.value,
                    "failure_count": self.metrics.current_failure_count
                }
            )
            raise
    
    async def _should_allow_call(self) -> bool:
        """Determine if the call should be allowed"""
        async with self._lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time and (current_time - self._last_failure_time >= self._current_timeout):
                    await self._transition_to_half_open()
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.metrics.current_success_count += 1
                if self.metrics.current_success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            else:
                # Reset failure count on success in CLOSED state
                self._cleanup_old_failures(time.time())
            
            # Emit metrics updated event periodically
            if self.metrics.total_requests % 10 == 0:  # Every 10 requests
                self._emit_event(
                    CircuitBreakerEventType.METRICS_UPDATED,
                    total_requests=self.metrics.total_requests,
                    success_rate=self.metrics.success_rate,
                    failure_rate=self.metrics.failure_rate
                )
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            current_time = time.time()
            
            # Track failure time for sliding window
            self._failure_times.append(current_time)
            self._last_failure_time = current_time
            self.metrics.current_failure_count = len(self._failure_times)
            
            # Clean old failures outside monitoring window
            self._cleanup_old_failures(current_time)
            
            # Check if circuit should open
            if self.metrics.current_failure_count >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    await self._transition_to_open()
    
    def _cleanup_old_failures(self, current_time: float):
        """Remove failures outside the monitoring window"""
        cutoff_time = current_time - self.config.monitoring_window
        self._failure_times = [t for t in self._failure_times if t > cutoff_time]
        self.metrics.current_failure_count = len(self._failure_times)
    
    async def _transition_to_half_open(self):
        """Transition to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.current_success_count = 0
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        self._invoke_callbacks(CircuitState.HALF_OPEN, old_state)
        self._emit_event(
            CircuitBreakerEventType.CIRCUIT_HALF_OPENED,
            old_state=old_state.value,
            new_state=self.state.value,
            failure_count=self.metrics.current_failure_count
        )
        self._emit_event(
            CircuitBreakerEventType.STATE_CHANGED,
            old_state=old_state.value,
            new_state=self.state.value
        )
    
    async def _transition_to_open(self):
        """Transition to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        logger.warning(
            f"Circuit breaker {self.name} OPENED after {self.metrics.current_failure_count} failures"
        )
        
        # Update adaptive timeout if enabled
        if self.config.enable_adaptive_timeout:
            self._update_adaptive_timeout()
        
        self._invoke_callbacks(CircuitState.OPEN, old_state)
        self._emit_event(
            CircuitBreakerEventType.CIRCUIT_OPENED,
            old_state=old_state.value,
            new_state=self.state.value,
            failure_count=self.metrics.current_failure_count,
            failure_threshold=self.config.failure_threshold
        )
        self._emit_event(
            CircuitBreakerEventType.STATE_CHANGED,
            old_state=old_state.value,
            new_state=self.state.value
        )
        self._emit_event(
            CircuitBreakerEventType.THRESHOLD_EXCEEDED,
            failure_count=self.metrics.current_failure_count,
            threshold=self.config.failure_threshold
        )
    
    async def _transition_to_closed(self):
        """Transition to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._failure_times.clear()
        self.metrics.current_failure_count = 0
        self.metrics.current_success_count = 0
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
        self._invoke_callbacks(CircuitState.CLOSED, old_state)
        self._emit_event(
            CircuitBreakerEventType.CIRCUIT_CLOSED,
            old_state=old_state.value,
            new_state=self.state.value,
            success_count=self.metrics.current_success_count
        )
        self._emit_event(
            CircuitBreakerEventType.STATE_CHANGED,
            old_state=old_state.value,
            new_state=self.state.value
        )
    
    def _update_adaptive_timeout(self):
        """Update timeout using enhanced adaptive algorithm"""
        if self.config.enable_adaptive_timeout:
            # If we have recent successes, reduce timeout gradually
            if self.metrics.successful_requests > 0:
                success_rate = self.metrics.success_rate
                if success_rate > 0.9:  # 90% success rate
                    # Reduce timeout gradually
                    new_timeout = max(
                        self.config.min_timeout,
                        self._current_timeout * 0.9
                    )
                    self._current_timeout = new_timeout
                    logger.info(f"Circuit breaker {self.name} adaptive timeout reduced to {new_timeout:.1f}s (success_rate: {success_rate:.2%})")
                    return
            
            # Increase timeout on failures
            new_timeout = min(
                self.config.max_timeout,
                max(
                    self.config.min_timeout,
                    self._current_timeout * self.config.timeout_multiplier
                )
            )
            self._current_timeout = new_timeout
            logger.info(f"Circuit breaker {self.name} adaptive timeout increased to {new_timeout:.1f}s")
    
    def _get_remaining_timeout(self) -> float:
        """Get remaining time until circuit can attempt recovery"""
        if self.state != CircuitState.OPEN or not self._last_failure_time:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self._current_timeout - elapsed)
    
    def _invoke_callbacks(self, new_state: CircuitState, old_state: CircuitState):
        """Invoke state transition callbacks"""
        for callback in self._state_callbacks.get(new_state, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback
                    asyncio.create_task(callback(self, new_state, old_state))
                else:
                    callback(self, new_state, old_state)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
    
    def _emit_event(self, event_type: CircuitBreakerEventType, **metadata):
        """Emit circuit breaker domain event"""
        event = CircuitBreakerEvent(
            event_type=event_type,
            circuit_name=self.name,
            old_state=self.state if event_type != CircuitBreakerEventType.STATE_CHANGED else None,
            metadata=metadata
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_event_history:
            self._event_history.pop(0)
        
        # Invoke event handlers
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in circuit breaker event handler: {e}")
    
    def on_event(self, handler: Callable[[CircuitBreakerEvent], None]):
        """
        Register event handler for circuit breaker events.
        
        Args:
            handler: Function to handle events (can be async or sync)
        """
        self._event_handlers.append(handler)
    
    def get_event_history(self, limit: Optional[int] = None) -> List[CircuitBreakerEvent]:
        """
        Get event history.
        
        Args:
            limit: Maximum number of events to return (default: all)
            
        Returns:
            List of events, most recent first
        """
        events = list(reversed(self._event_history))
        if limit:
            return events[:limit]
        return events
    
    def get_events_by_type(self, event_type: CircuitBreakerEventType, limit: Optional[int] = None) -> List[CircuitBreakerEvent]:
        """
        Get events filtered by type.
        
        Args:
            event_type: Type of events to filter
            limit: Maximum number of events to return
            
        Returns:
            List of filtered events
        """
        events = [e for e in reversed(self._event_history) if e.event_type == event_type]
        if limit:
            return events[:limit]
        return events
    
    def on_state_change(self, state: CircuitState, callback: Callable):
        """Register callback for state transitions"""
        self._state_callbacks[state].append(callback)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.metrics.current_failure_count,
            "success_count": self.metrics.current_success_count,
            "last_failure_time": datetime.fromtimestamp(self._last_failure_time).isoformat() if self._last_failure_time else None,
            "metrics": self.metrics.to_dict(),
            "health": self.get_health_status(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "monitoring_window": self.config.monitoring_window,
                "call_timeout": self.config.call_timeout,
                "retry_enabled": self.config.retry_enabled,
                "fallback_enabled": self.config.fallback_enabled,
                "half_open_max_concurrent": self.config.half_open_max_concurrent,
            },
            "remaining_timeout": self._get_remaining_timeout()
        }
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics"""
        return self.metrics
    
    async def reset(self):
        """Manually reset circuit breaker"""
        async with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self._failure_times.clear()
            self._last_failure_time = None
            self._current_timeout = self.config.recovery_timeout
            logger.info(f"Circuit breaker {self.name} manually reset")
            self._invoke_callbacks(CircuitState.CLOSED, old_state)
    
    def force_open(self):
        """Force circuit breaker to open state"""
        async def _force_open():
            async with self._lock:
                if self.state != CircuitState.OPEN:
                    await self._transition_to_open()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_force_open())
            else:
                loop.run_until_complete(_force_open())
        except RuntimeError:
            # No event loop, just update state
            self.state = CircuitState.OPEN
    
    def force_close(self):
        """Force circuit breaker to closed state"""
        async def _force_close():
            async with self._lock:
                if self.state != CircuitState.CLOSED:
                    await self._transition_to_closed()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_force_close())
            else:
                loop.run_until_complete(_force_close())
        except RuntimeError:
            # No event loop, just update state
            self.state = CircuitState.CLOSED
    
    # Context manager support
    async def __aenter__(self):
        """
        Async context manager entry.
        
        Returns:
            CircuitBreaker instance
            
        Example:
            async with breaker:
                result = await breaker.call(func, ...)
        """
        # Emit event for context entry
        self._emit_event(
            CircuitBreakerEventType.METRICS_UPDATED,
            action="context_entered",
            state=self.state.value
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            False to propagate exceptions, True to suppress them
        """
        # Emit event for context exit
        self._emit_event(
            CircuitBreakerEventType.METRICS_UPDATED,
            action="context_exited",
            state=self.state.value,
            exception_occurred=exc_type is not None,
            exception_type=exc_type.__name__ if exc_type else None
        )
        
        # Optionally reset on exit if configured
        if self.config.auto_reset_on_exit:
            await self.reset()
            logger.debug(f"Circuit breaker {self.name} auto-reset on context exit")
        
        # Return False to propagate exceptions
        return False
    
    # Health check methods
    def is_healthy(self) -> bool:
        """
        Check if circuit breaker is in healthy state.
        
        Returns:
            True if circuit is CLOSED or HALF_OPEN with recent successes
        """
        if self.state == CircuitState.CLOSED:
            # Check success rate threshold if we have enough requests
            if self.metrics.total_requests >= 10:
                return self.metrics.success_rate >= self.config.health_success_rate_threshold
            return True
        elif self.state == CircuitState.HALF_OPEN:
            # Consider healthy if we have at least one success
            return self.metrics.current_success_count > 0
        return False
    
    def is_ready(self) -> bool:
        """
        Check if circuit breaker is ready to accept requests.
        
        Returns:
            True if circuit is not OPEN
        """
        return self.state != CircuitState.OPEN
    
    def is_degraded(self) -> bool:
        """
        Check if circuit breaker is in degraded state.
        
        Degraded means operational but with reduced performance/reliability.
        
        Returns:
            True if circuit is operational but success rate is below threshold
        """
        if self.state == CircuitState.OPEN:
            return False  # Not degraded, just unavailable
        
        if self.metrics.total_requests < 10:
            return False  # Not enough data
        
        success_rate = self.metrics.success_rate
        return (
            success_rate < self.config.health_success_rate_threshold and
            success_rate >= self.config.health_degraded_threshold
        )
    
    def is_critical(self) -> bool:
        """
        Check if circuit breaker is in critical state.
        
        Critical means very low success rate or approaching failure threshold.
        
        Returns:
            True if circuit is in critical condition
        """
        if self.state == CircuitState.OPEN:
            return True
        
        if self.metrics.total_requests < 5:
            return False  # Not enough data
        
        # Critical if success rate is very low
        if self.metrics.success_rate < self.config.health_degraded_threshold:
            return True
        
        # Critical if approaching failure threshold
        if (self.metrics.current_failure_count >= 
            self.config.failure_threshold * 0.8):  # 80% of threshold
            return True
        
        return False
    
    def get_health_score(self) -> float:
        """
        Get health score from 0.0 to 1.0.
        
        Returns:
            Health score where 1.0 is perfect health, 0.0 is critical
        """
        if self.state == CircuitState.OPEN:
            return 0.0
        
        if self.metrics.total_requests == 0:
            return 1.0  # No data, assume healthy
        
        # Base score on success rate
        base_score = self.metrics.success_rate
        
        # Adjust based on state
        if self.state == CircuitState.HALF_OPEN:
            # Reduce score slightly for half-open
            base_score *= 0.8
        
        # Penalize if approaching failure threshold
        failure_ratio = self.metrics.current_failure_count / self.config.failure_threshold
        if failure_ratio > 0.5:
            penalty = (failure_ratio - 0.5) * 0.4  # Up to 20% penalty
            base_score *= (1.0 - penalty)
        
        return max(0.0, min(1.0, base_score))
    
    def get_health_rating(self) -> str:
        """
        Get human-readable health rating.
        
        Returns:
            One of: "excellent", "good", "degraded", "critical", "unavailable"
        """
        if self.state == CircuitState.OPEN:
            return "unavailable"
        
        score = self.get_health_score()
        
        if score >= 0.95:
            return "excellent"
        elif score >= 0.80:
            return "good"
        elif score >= 0.60:
            return "degraded"
        elif score >= 0.30:
            return "critical"
        else:
            return "critical"
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status.
        
        Returns:
            Dictionary with comprehensive health information
        """
        health_score = self.get_health_score()
        health_rating = self.get_health_rating()
        
        return {
            "healthy": self.is_healthy(),
            "ready": self.is_ready(),
            "degraded": self.is_degraded(),
            "critical": self.is_critical(),
            "health_score": health_score,
            "health_rating": health_rating,
            "state": self.state.value,
            "failure_count": self.metrics.current_failure_count,
            "success_count": self.metrics.current_success_count,
            "success_rate": self.metrics.success_rate,
            "failure_rate": self.metrics.failure_rate,
            "total_requests": self.metrics.total_requests,
            "remaining_timeout": self._get_remaining_timeout(),
            "thresholds": {
                "failure_threshold": self.config.failure_threshold,
                "success_rate_threshold": self.config.health_success_rate_threshold,
                "degraded_threshold": self.config.health_degraded_threshold
            },
            "recommendations": self._get_health_recommendations()
        }
    
    def _get_health_recommendations(self) -> List[str]:
        """
        Get health recommendations based on current state.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if self.state == CircuitState.OPEN:
            recommendations.append("Circuit is open. Wait for recovery timeout or manually reset.")
            recommendations.append(f"Recovery in {self._get_remaining_timeout():.1f}s")
        
        elif self.is_critical():
            recommendations.append("Circuit is in critical state. Monitor closely.")
            if self.metrics.current_failure_count > 0:
                recommendations.append(
                    f"Current failure count: {self.metrics.current_failure_count}/"
                    f"{self.config.failure_threshold}"
                )
        
        elif self.is_degraded():
            recommendations.append("Circuit is degraded. Success rate below optimal threshold.")
            recommendations.append(f"Current success rate: {self.metrics.success_rate:.2%}")
        
        if self.metrics.total_requests < 10:
            recommendations.append("Insufficient data for accurate health assessment.")
        
        if self.metrics.failure_rate > 0.5:
            recommendations.append("High failure rate detected. Investigate service issues.")
        
        if not recommendations:
            recommendations.append("Circuit breaker is operating normally.")
        
        return recommendations
    
    async def call_with_fallback(
        self,
        func: Callable,
        fallback_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with fallback when circuit is open.
        
        Args:
            func: Primary function to call
            fallback_func: Fallback function to call if circuit is open
            *args: Arguments for functions
            **kwargs: Keyword arguments for functions
            
        Returns:
            Result from primary function or fallback function
        """
        try:
            return await self.call(func, *args, **kwargs)
        except ServiceError as e:
            # Only use fallback if circuit is open
            if self.state == CircuitState.OPEN:
                self.metrics.fallback_count += 1
                self._emit_event(
                    CircuitBreakerEventType.FALLBACK_USED,
                    state=self.state.value
                )
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            raise
    
    async def call_bulk(
        self,
        func: Callable,
        items: List[Tuple],
        stop_on_first_error: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        Process multiple calls in bulk with circuit breaker protection.
        
        Args:
            func: Function to call for each item
            items: List of tuples, each tuple contains args for one call
            stop_on_first_error: If True, stop processing on first error
            **kwargs: Common keyword arguments for all calls
            
        Returns:
            List of results (None for failed calls if stop_on_first_error=False)
            
        Example:
            items = [("arg1", "arg2"), ("arg3", "arg4")]
            results = await breaker.call_bulk(api_call, items)
        """
        results = []
        errors = []
        
        for i, item_args in enumerate(items):
            try:
                # Convert tuple to args
                if isinstance(item_args, tuple):
                    result = await self.call(func, *item_args, **kwargs)
                else:
                    result = await self.call(func, item_args, **kwargs)
                results.append(result)
            except Exception as e:
                errors.append((i, e))
                if stop_on_first_error:
                    # Re-raise first error
                    raise
                results.append(None)
        
        # Emit event for bulk operation
        self._emit_event(
            CircuitBreakerEventType.METRICS_UPDATED,
            action="bulk_operation",
            total_items=len(items),
            successful=len([r for r in results if r is not None]),
            failed=len(errors)
        )
        
        return results
    
    async def update_config(self, **updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update circuit breaker configuration dynamically.
        
        Args:
            **updates: Configuration parameters to update
            
        Returns:
            Dictionary with updated configuration
            
        Example:
            await breaker.update_config(
                failure_threshold=10,
                recovery_timeout=120.0
            )
        """
        async with self._lock:
            updated = {}
            
            # Validate and update configuration
            for key, value in updates.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    updated[key] = {"old": old_value, "new": value}
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
            
            # Update internal state if needed
            if "recovery_timeout" in updates:
                self._current_timeout = self.config.recovery_timeout
            
            # Emit event
            self._emit_event(
                CircuitBreakerEventType.METRICS_UPDATED,
                action="config_updated",
                updates=updated
            )
            
            logger.info(
                f"Circuit breaker {self.name} configuration updated",
                extra={"updates": updated}
            )
            
            return updated
    
    def export_metrics_prometheus(self) -> Dict[str, Any]:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Dictionary with Prometheus-formatted metrics
        """
        metrics = self.metrics.to_dict()
        
        return {
            "circuit_breaker_state": {
                "value": 1.0 if self.state == CircuitState.CLOSED else 0.0,
                "labels": {
                    "circuit_name": self.name,
                    "state": self.state.value
                }
            },
            "circuit_breaker_requests_total": {
                "value": float(metrics["total_requests"]),
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_requests_successful": {
                "value": float(metrics["successful_requests"]),
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_requests_failed": {
                "value": float(metrics["failed_requests"]),
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_requests_rejected": {
                "value": float(metrics["rejected_requests"]),
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_success_rate": {
                "value": metrics["success_rate"],
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_failure_rate": {
                "value": metrics["failure_rate"],
                "labels": {"circuit_name": self.name}
            },
            "circuit_breaker_response_time_seconds": {
                "value": metrics.get("avg_response_time", 0.0),
                "labels": {"circuit_name": self.name, "quantile": "avg"}
            },
            "circuit_breaker_response_time_p50": {
                "value": metrics.get("p50_response_time", 0.0),
                "labels": {"circuit_name": self.name, "quantile": "0.5"}
            },
            "circuit_breaker_response_time_p95": {
                "value": metrics.get("p95_response_time", 0.0),
                "labels": {"circuit_name": self.name, "quantile": "0.95"}
            },
            "circuit_breaker_response_time_p99": {
                "value": metrics.get("p99_response_time", 0.0),
                "labels": {"circuit_name": self.name, "quantile": "0.99"}
            },
            "circuit_breaker_health_score": {
                "value": self.get_health_score(),
                "labels": {"circuit_name": self.name}
            }
        }
    
    def export_metrics_statsd(self) -> List[Dict[str, Any]]:
        """
        Export metrics in StatsD format.
        
        Returns:
            List of dictionaries with StatsD-formatted metrics
        """
        metrics = self.metrics.to_dict()
        base_name = f"circuit_breaker.{self.name}"
        
        return [
            {
                "type": "gauge",
                "name": f"{base_name}.state",
                "value": 1.0 if self.state == CircuitState.CLOSED else 0.0,
                "tags": {"state": self.state.value}
            },
            {
                "type": "counter",
                "name": f"{base_name}.requests.total",
                "value": metrics["total_requests"]
            },
            {
                "type": "counter",
                "name": f"{base_name}.requests.successful",
                "value": metrics["successful_requests"]
            },
            {
                "type": "counter",
                "name": f"{base_name}.requests.failed",
                "value": metrics["failed_requests"]
            },
            {
                "type": "counter",
                "name": f"{base_name}.requests.rejected",
                "value": metrics["rejected_requests"]
            },
            {
                "type": "gauge",
                "name": f"{base_name}.success_rate",
                "value": metrics["success_rate"]
            },
            {
                "type": "gauge",
                "name": f"{base_name}.failure_rate",
                "value": metrics["failure_rate"]
            },
            {
                "type": "histogram",
                "name": f"{base_name}.response_time",
                "value": metrics.get("avg_response_time", 0.0)
            },
            {
                "type": "gauge",
                "name": f"{base_name}.health_score",
                "value": self.get_health_score()
            }
        ]
