"""
Advanced error handling and recovery mechanisms for Blaze AI.

This module provides robust error handling, circuit breaker patterns,
retry mechanisms, and graceful degradation strategies.
"""

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from contextlib import asynccontextmanager

from ..core.interfaces import LogLevel

# =============================================================================
# Types
# =============================================================================

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# =============================================================================
# Enums
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

# =============================================================================
# Error Classes
# =============================================================================

class BlazeAIError(Exception):
    """Base exception for Blaze AI module."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class ServiceUnavailableError(BlazeAIError):
    """Raised when a service is unavailable."""
    pass

class RateLimitExceededError(BlazeAIError):
    """Raised when rate limits are exceeded."""
    pass

class ValidationError(BlazeAIError):
    """Raised when input validation fails."""
    pass

# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    monitor_interval: float = 10.0
    max_failures: int = 10

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        self._lock = asyncio.Lock()
        
    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True
    
    async def on_success(self):
        """Handle successful execution."""
        async with self._lock:
            self.failure_count = 0
            self.last_success_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
    
    async def on_failure(self, exception: Exception):
        """Handle execution failure."""
        async with self._lock:
            if isinstance(exception, self.config.expected_exception):
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "can_execute": self.state != CircuitState.OPEN
        }

# =============================================================================
# Retry Mechanisms
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)

class RetryHandler:
    """Retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if self.config.retryable_exceptions and not any(
                    isinstance(e, exc_type) for exc_type in self.config.retryable_exceptions
                ):
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                # Add jitter if enabled
                if self.config.jitter:
                    delay *= (0.5 + 0.5 * time.time() % 1)
                
                await asyncio.sleep(delay)
        
        raise last_exception

# =============================================================================
# Error Recovery Strategies
# =============================================================================

class ErrorRecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies."""
    
    @abstractmethod
    async def can_recover(self, error: Exception) -> bool:
        """Check if error can be recovered from."""
        pass
    
    @abstractmethod
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from error."""
        pass

class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_func: Callable[..., Any]):
        self.fallback_func = fallback_func
    
    async def can_recover(self, error: Exception) -> bool:
        return True
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(self.fallback_func):
            return await self.fallback_func(**context)
        return self.fallback_func(**context)

class DegradationStrategy(ErrorRecoveryStrategy):
    """Graceful degradation strategy."""
    
    def __init__(self, degraded_features: Dict[str, Any]):
        self.degraded_features = degraded_features
    
    async def can_recover(self, error: Exception) -> bool:
        return True
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        feature = context.get('feature', 'default')
        return self.degraded_features.get(feature, None)

# =============================================================================
# Decorators
# =============================================================================

def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to apply circuit breaker pattern."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not await circuit_breaker.can_execute():
                raise ServiceUnavailableError(
                    f"Service {circuit_breaker.name} is currently unavailable"
                )
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await circuit_breaker.on_success()
                return result
            except Exception as e:
                await circuit_breaker.on_failure(e)
                raise
        
        return wrapper
    return decorator

def with_retry(config: RetryConfig):
    """Decorator to apply retry logic."""
    def decorator(func: F) -> F:
        retry_handler = RetryHandler(config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    return decorator

def with_error_recovery(strategy: ErrorRecoveryStrategy):
    """Decorator to apply error recovery strategy."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            except Exception as e:
                if await strategy.can_recover(e):
                    context = {"args": args, "kwargs": kwargs}
                    return await strategy.recover(e, context)
                raise
        
        return wrapper
    return decorator

# =============================================================================
# Error Monitoring
# =============================================================================

class ErrorMonitor:
    """Monitor and track errors for analysis."""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.severity_counts: Dict[ErrorSeverity, int] = {}
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record an error for monitoring."""
        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "message": str(error),
            "severity": getattr(error, 'severity', ErrorSeverity.MEDIUM),
            "context": context or {}
        }
        
        self.errors.append(error_info)
        self.error_counts[error_info["error_type"]] = self.error_counts.get(error_info["error_type"], 0) + 1
        self.severity_counts[error_info["severity"]] = self.severity_counts.get(error_info["severity"], 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "severity_counts": {k.value: v for k, v in self.severity_counts.items()},
            "recent_errors": self.errors[-10:] if self.errors else []
        }
    
    def clear_errors(self):
        """Clear recorded errors."""
        self.errors.clear()
        self.error_counts.clear()
        self.severity_counts.clear()

# =============================================================================
# Global Error Monitor Instance
# =============================================================================

error_monitor = ErrorMonitor()

def get_error_monitor() -> ErrorMonitor:
    """Get global error monitor instance."""
    return error_monitor
