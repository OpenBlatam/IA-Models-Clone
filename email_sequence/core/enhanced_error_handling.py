"""
Enhanced Error Handling and Resilience System

Provides comprehensive error handling, circuit breakers, retry mechanisms,
and resilience patterns for the email sequence system.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import functools
import json
import hashlib

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 60
DEFAULT_TIMEOUT = 30


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str
    timestamp: datetime
    operation: str
    component: str
    severity: ErrorSeverity
    error_type: str
    error_message: str
    stack_trace: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    recovery_timeout: int = DEFAULT_CIRCUIT_BREAKER_TIMEOUT
    expected_exception: Type[Exception] = Exception
    monitor_interval: int = 10
    enable_monitoring: bool = True
    enable_metrics: bool = True


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True
    timeout: float = DEFAULT_TIMEOUT
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class ResilienceMetrics:
    """Resilience system metrics"""
    total_errors: int = 0
    handled_errors: int = 0
    unhandled_errors: int = 0
    retry_attempts: int = 0
    successful_retries: int = 0
    circuit_breaker_trips: int = 0
    circuit_breaker_resets: int = 0
    avg_error_handling_time: float = 0.0
    error_rate: float = 0.0


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.monitoring_task = None
        self.is_monitoring = False
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
    
    def _on_failure(self, error: Exception):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    async def start_monitoring(self):
        """Start circuit breaker monitoring"""
        if not self.config.enable_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Circuit breaker '{self.name}' monitoring started")
    
    async def stop_monitoring(self):
        """Stop circuit breaker monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Monitoring loop for circuit breaker"""
        while self.is_monitoring:
            try:
                # Log circuit breaker state
                logger.debug(f"Circuit breaker '{self.name}' state: {self.state.value}")
                
                # Check for automatic reset
                if self.state == CircuitBreakerState.OPEN and self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' automatically moved to HALF_OPEN")
                
                await asyncio.sleep(self.config.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {e}")
                await asyncio.sleep(1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout
        }


class RetryHandler:
    """Retry mechanism with configurable strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.retry_attempts = 0
        self.successful_retries = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry mechanism"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retry attempt {attempt}/{self.config.max_retries} after {delay:.2f}s")
                    await asyncio.sleep(delay)
                
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                
                if attempt > 0:
                    self.successful_retries += 1
                    logger.info(f"Retry successful on attempt {attempt}")
                
                return result
                
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(f"Operation timed out after {self.config.timeout}s")
                logger.warning(f"Operation timed out on attempt {attempt + 1}")
                
            except Exception as e:
                last_exception = e
                self.retry_attempts += 1
                
                if not self._should_retry(e):
                    logger.error(f"Non-retryable error on attempt {attempt + 1}: {e}")
                    break
                
                logger.warning(f"Retryable error on attempt {attempt + 1}: {e}")
        
        # All retries exhausted
        logger.error(f"All retry attempts exhausted for operation")
        raise last_exception
    
    def _should_retry(self, error: Exception) -> bool:
        """Check if error should be retried"""
        return isinstance(error, self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.retry_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.retry_delay * (self.config.backoff_factor ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.retry_delay * attempt
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.retry_delay * self._fibonacci(attempt)
        else:
            delay = self.config.retry_delay
        
        # Apply jitter
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return min(delay, self.config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics"""
        return {
            "retry_attempts": self.retry_attempts,
            "successful_retries": self.successful_retries,
            "success_rate": self.successful_retries / max(self.retry_attempts, 1)
        }


class ErrorTracker:
    """Comprehensive error tracking and analysis"""
    
    def __init__(self):
        self.errors: List[ErrorContext] = []
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.component_errors: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=1000)
        
        # Error handlers
        self.error_handlers: Dict[ErrorSeverity, List[Callable]] = defaultdict(list)
        self.global_error_handlers: List[Callable] = []
        
        logger.info("Error tracker initialized")
    
    def track_error(
        self,
        error: Exception,
        operation: str,
        component: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track an error with context"""
        error_id = self._generate_error_id()
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            operation=operation,
            component=component,
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        # Store error
        self.errors.append(error_context)
        self.recent_errors.append(error_context)
        self.component_errors[component].append(error_context)
        self.severity_counts[severity] += 1
        
        # Track error pattern
        pattern_key = f"{component}:{type(error).__name__}"
        self.error_patterns[pattern_key] += 1
        
        # Trigger error handlers
        asyncio.create_task(self._trigger_error_handlers(error_context))
        
        logger.error(f"Error tracked: {error_id} - {error} in {component}.{operation}")
        return error_id
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        timestamp = datetime.utcnow().isoformat()
        random_suffix = hashlib.md5(f"{timestamp}{len(self.errors)}".encode()).hexdigest()[:8]
        return f"err_{timestamp.replace(':', '-').replace('.', '-')}_{random_suffix}"
    
    async def _trigger_error_handlers(self, error_context: ErrorContext):
        """Trigger error handlers"""
        # Global handlers
        for handler in self.global_error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_context)
                else:
                    handler(error_context)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
        
        # Severity-specific handlers
        for handler in self.error_handlers[error_context.severity]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_context)
                else:
                    handler(error_context)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
    
    def add_error_handler(self, handler: Callable, severity: Optional[ErrorSeverity] = None):
        """Add error handler"""
        if severity is None:
            self.global_error_handlers.append(handler)
        else:
            self.error_handlers[severity].append(handler)
    
    def get_error_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error analytics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.errors if e.timestamp > cutoff_time]
        
        analytics = {
            "total_errors": len(recent_errors),
            "errors_by_severity": {},
            "errors_by_component": {},
            "errors_by_type": {},
            "error_rate_per_hour": len(recent_errors) / hours,
            "most_common_errors": [],
            "recent_error_trend": []
        }
        
        # Errors by severity
        for severity in ErrorSeverity:
            count = len([e for e in recent_errors if e.severity == severity])
            analytics["errors_by_severity"][severity.value] = count
        
        # Errors by component
        component_counts = defaultdict(int)
        for error in recent_errors:
            component_counts[error.component] += 1
        analytics["errors_by_component"] = dict(component_counts)
        
        # Errors by type
        type_counts = defaultdict(int)
        for error in recent_errors:
            type_counts[error.error_type] += 1
        analytics["errors_by_type"] = dict(type_counts)
        
        # Most common errors
        pattern_counts = defaultdict(int)
        for error in recent_errors:
            pattern = f"{error.component}:{error.error_type}"
            pattern_counts[pattern] += 1
        
        analytics["most_common_errors"] = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return analytics
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorContext]:
        """Get error by ID"""
        for error in self.errors:
            if error.error_id == error_id:
                return error
        return None
    
    def clear_old_errors(self, days: int = 30):
        """Clear errors older than specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        self.errors = [e for e in self.errors if e.timestamp > cutoff_time]
        
        # Rebuild component errors
        self.component_errors.clear()
        for error in self.errors:
            self.component_errors[error.component].append(error)
        
        logger.info(f"Cleared errors older than {days} days")


class ResilienceManager:
    """Main resilience manager coordinating all resilience features"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.error_tracker = ErrorTracker()
        self.metrics = ResilienceMetrics()
        
        # Configuration
        self.default_retry_config = RetryConfig()
        self.default_circuit_breaker_config = CircuitBreakerConfig()
        
        logger.info("Resilience manager initialized")
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create a circuit breaker"""
        if config is None:
            config = self.default_circuit_breaker_config
        
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        
        return circuit_breaker
    
    def create_retry_handler(self, name: str, config: Optional[RetryConfig] = None) -> RetryHandler:
        """Create a retry handler"""
        if config is None:
            config = self.default_retry_config
        
        retry_handler = RetryHandler(config)
        self.retry_handlers[name] = retry_handler
        
        return retry_handler
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable,
        *args,
        circuit_breaker_name: Optional[str] = None,
        retry_handler_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute function with full resilience protection"""
        start_time = time.time()
        
        try:
            # Get circuit breaker
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = self.circuit_breakers.get(circuit_breaker_name)
            
            # Get retry handler
            retry_handler = None
            if retry_handler_name:
                retry_handler = self.retry_handlers.get(retry_handler_name)
            
            # Execute with circuit breaker
            if circuit_breaker:
                if retry_handler:
                    # Use retry handler inside circuit breaker
                    result = await circuit_breaker.call(
                        lambda: retry_handler.execute(func, *args, **kwargs)
                    )
                else:
                    # Use circuit breaker only
                    result = await circuit_breaker.call(func, *args, **kwargs)
            else:
                if retry_handler:
                    # Use retry handler only
                    result = await retry_handler.execute(func, *args, **kwargs)
                else:
                    # No resilience protection
                    result = await func(*args, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics.handled_errors += 1
            self.metrics.avg_error_handling_time = (
                (self.metrics.avg_error_handling_time * (self.metrics.handled_errors - 1) + execution_time) /
                self.metrics.handled_errors
            )
            
            return result
            
        except Exception as e:
            # Track error
            error_id = self.error_tracker.track_error(
                error=e,
                operation=operation_name,
                component="resilience_manager",
                severity=ErrorSeverity.HIGH
            )
            
            # Update metrics
            self.metrics.total_errors += 1
            self.metrics.unhandled_errors += 1
            
            # Re-raise the exception
            raise
    
    def add_error_handler(self, handler: Callable, severity: Optional[ErrorSeverity] = None):
        """Add error handler to the error tracker"""
        self.error_tracker.add_error_handler(handler, severity)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        circuit_breaker_metrics = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_metrics[name] = cb.get_metrics()
        
        retry_metrics = {}
        for name, rh in self.retry_handlers.items():
            retry_metrics[name] = rh.get_metrics()
        
        return {
            "resilience_metrics": {
                "total_errors": self.metrics.total_errors,
                "handled_errors": self.metrics.handled_errors,
                "unhandled_errors": self.metrics.unhandled_errors,
                "retry_attempts": self.metrics.retry_attempts,
                "successful_retries": self.metrics.successful_retries,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "circuit_breaker_resets": self.metrics.circuit_breaker_resets,
                "avg_error_handling_time": self.metrics.avg_error_handling_time,
                "error_rate": self.metrics.error_rate
            },
            "circuit_breakers": circuit_breaker_metrics,
            "retry_handlers": retry_metrics,
            "error_analytics": self.error_tracker.get_error_analytics()
        }
    
    async def start_monitoring(self):
        """Start monitoring for all circuit breakers"""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.start_monitoring()
    
    async def stop_monitoring(self):
        """Stop monitoring for all circuit breakers"""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.stop_monitoring()


# Decorators for easy resilience
def resilient(
    circuit_breaker_name: Optional[str] = None,
    retry_handler_name: Optional[str] = None,
    operation_name: Optional[str] = None
):
    """Decorator for adding resilience to functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get resilience manager (you might want to make this globally accessible)
            resilience_manager = getattr(wrapper, '_resilience_manager', None)
            if resilience_manager is None:
                # Fallback to direct execution
                return await func(*args, **kwargs)
            
            op_name = operation_name or func.__name__
            return await resilience_manager.execute_with_resilience(
                op_name,
                func,
                *args,
                circuit_breaker_name=circuit_breaker_name,
                retry_handler_name=retry_handler_name,
                **kwargs
            )
        
        return wrapper
    return decorator


def error_handler(severity: Optional[ErrorSeverity] = None):
    """Decorator for registering error handlers"""
    def decorator(handler_func):
        # This would register the handler with the global error tracker
        # You might want to make the error tracker globally accessible
        return handler_func
    return decorator


# Global resilience manager instance
_resilience_manager = None

def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager"""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager 