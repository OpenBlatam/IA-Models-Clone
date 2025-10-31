"""
Enhanced Error Handling Module for Blaze AI.

This module provides comprehensive error handling, circuit breaker patterns,
retry mechanisms, and error recovery strategies.
"""

import asyncio
import functools
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import WeakKeyDictionary

from core.config import ErrorHandlingConfig
from core.exceptions import BlazeAIError, ServiceUnavailableError
from core.logging import get_logger


# ============================================================================
# ERROR HANDLING MODELS AND CONFIGURATION
# ============================================================================

class ErrorRecord(BaseModel):
    """Record of an error occurrence."""
    error_id: str = Field(default_factory=lambda: str(time.time()))
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    severity: str = Field(default="medium", description="Error severity")
    retry_count: int = Field(0, description="Number of retry attempts")
    resolved: bool = Field(False, description="Whether error was resolved")


class CircuitBreakerState(BaseModel):
    """Circuit breaker state information."""
    state: str = Field(..., description="Current state (closed, open, half_open)")
    failure_count: int = Field(0, description="Consecutive failure count")
    last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
    last_success_time: Optional[datetime] = Field(None, description="Last success timestamp")
    next_attempt_time: Optional[datetime] = Field(None, description="Next attempt timestamp")
    total_requests: int = Field(0, description="Total requests processed")
    successful_requests: int = Field(0, description="Successful requests")


class RetryStrategy(BaseModel):
    """Configuration for retry strategies."""
    max_attempts: int = Field(3, description="Maximum retry attempts")
    base_delay: float = Field(1.0, description="Base delay in seconds")
    max_delay: float = Field(60.0, description="Maximum delay in seconds")
    backoff_factor: float = Field(2.0, description="Exponential backoff factor")
    jitter: bool = Field(True, description="Add random jitter to delays")
    retryable_errors: List[Type[Exception]] = Field(default_factory=list, description="Retryable error types")


# ============================================================================
# ERROR HANDLING INTERFACES AND BASE CLASSES
# ============================================================================

class ErrorHandler(ABC):
    """Abstract base class for error handlers."""
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle an error occurrence."""
        pass
    
    @abstractmethod
    async def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        pass


class CircuitBreaker(ABC):
    """Abstract base class for circuit breaker implementations."""
    
    @abstractmethod
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        pass
    
    @abstractmethod
    async def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        pass


class RetryHandler(ABC):
    """Abstract base class for retry handlers."""
    
    @abstractmethod
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        pass


class ErrorRecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies."""
    
    @abstractmethod
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error."""
        pass


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

class ErrorHandlerImpl(ErrorHandler):
    """Implementation of error handling functionality."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.errors: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle an error occurrence."""
        try:
            # Create error record
            error_record = ErrorRecord(
                error_type=type(error).__name__,
                error_message=str(error),
                context=context,
                severity=self._determine_severity(error)
            )
            
            # Add to error list
            self.errors.append(error_record)
            
            # Update counters
            self.error_counts[error_record.error_type] += 1
            self.severity_counts[error_record.severity] += 1
            
            # Log error
            self.logger.error(
                f"Error handled: {error_record.error_type} - {error_record.error_message}",
                extra={"error_context": context}
            )
            
            # Maintain error list size
            if len(self.errors) > 1000:
                self.errors = self.errors[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
    
    async def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        try:
            recent_errors = [
                e for e in self.errors
                if e.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            return {
                "total_errors": len(self.errors),
                "recent_errors": len(recent_errors),
                "error_types": dict(self.error_counts),
                "severity_distribution": dict(self.severity_counts),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            return {"error": str(e)}
    
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity level."""
        if isinstance(error, (ValueError, TypeError)):
            return "low"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return "medium"
        elif isinstance(error, (OSError, ServiceUnavailableError)):
            return "high"
        else:
            return "medium"


class CircuitBreakerImpl(CircuitBreaker):
    """Implementation of circuit breaker pattern."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.next_attempt_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit breaker is open
            if self.state == "open":
                if self.next_attempt_time and datetime.utcnow() < self.next_attempt_time:
                    raise ServiceUnavailableError(
                        f"Circuit breaker is open. Next attempt at {self.next_attempt_time}"
                    )
                else:
                    # Try to transition to half-open
                    self.state = "half_open"
                    self.logger.info("Circuit breaker transitioning to half-open state")
            
            # Execute function
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Success - close circuit breaker
                await self._on_success()
                return result
                
            except Exception as e:
                # Failure - handle according to circuit breaker state
                await self._on_failure(e)
                raise
    
    async def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return CircuitBreakerState(
            state=self.state,
            failure_count=self.failure_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            next_attempt_time=self.next_attempt_time,
            total_requests=self.total_requests,
            successful_requests=self.successful_requests
        )
    
    async def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.last_success_time = datetime.utcnow()
        self.successful_requests += 1
        self.total_requests += 1
        
        if self.state != "closed":
            self.state = "closed"
            self.logger.info("Circuit breaker closed after successful execution")
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle execution failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.total_requests += 1
        
        # Check if threshold reached
        if self.failure_count >= self.config.circuit_breaker_threshold:
            if self.state == "half_open":
                # Half-open state - open circuit breaker
                self.state = "open"
                self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.config.circuit_breaker_timeout)
                self.logger.warning("Circuit breaker opened after failure in half-open state")
            elif self.state == "closed":
                # Closed state - open circuit breaker
                self.state = "open"
                self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.config.circuit_breaker_timeout)
                self.logger.warning("Circuit breaker opened after reaching failure threshold")


class RetryHandlerImpl(RetryHandler):
    """Implementation of retry handling functionality."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - return result
                if attempt > 0:
                    self.logger.info(f"Function succeeded after {attempt} retries")
                return result
                
            except Exception as e:
                last_error = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    self.logger.warning(f"Non-retryable error encountered: {e}")
                    break
                
                # Check if max attempts reached
                if attempt >= self.config.max_retry_attempts:
                    self.logger.error(f"Max retry attempts ({self.config.max_retry_attempts}) reached")
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                self.logger.info(f"Retry attempt {attempt + 1} in {delay:.2f} seconds")
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        raise last_error or Exception("Unknown error in retry handler")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        # Check configured retryable errors
        for retryable_type in self.config.retryable_errors:
            if isinstance(error, retryable_type):
                return True
        
        # Check common retryable errors
        retryable_types = (ConnectionError, TimeoutError, OSError)
        return isinstance(error, retryable_types)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay


class FallbackRecoveryStrategy(ErrorRecoveryStrategy):
    """Fallback-based error recovery strategy."""
    
    def __init__(self, fallback_func: Optional[Callable] = None):
        self.fallback_func = fallback_func
        self.logger = get_logger(__name__)
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error using fallback."""
        try:
            if self.fallback_func:
                self.logger.info("Attempting recovery with fallback function")
                
                # Execute fallback function
                if asyncio.iscoroutinefunction(self.fallback_func):
                    result = await self.fallback_func(error, context)
                else:
                    result = self.fallback_func(error, context)
                
                self.logger.info("Recovery with fallback successful")
                return True
            else:
                self.logger.info("No fallback function available for recovery")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in fallback recovery: {e}")
            return False


class GracefulDegradationStrategy(ErrorRecoveryStrategy):
    """Graceful degradation error recovery strategy."""
    
    def __init__(self, degraded_services: Dict[str, Callable]):
        self.degraded_services = degraded_services
        self.logger = get_logger(__name__)
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover using graceful degradation."""
        try:
            service_name = context.get("service_name", "unknown")
            
            if service_name in self.degraded_services:
                self.logger.info(f"Attempting graceful degradation for service: {service_name}")
                
                # Execute degraded service
                degraded_func = self.degraded_services[service_name]
                if asyncio.iscoroutinefunction(degraded_func):
                    result = await degraded_func(error, context)
                else:
                    result = degraded_func(error, context)
                
                self.logger.info(f"Graceful degradation successful for service: {service_name}")
                return True
            else:
                self.logger.info(f"No degraded service available for: {service_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in graceful degradation: {e}")
            return False


# ============================================================================
# MAIN ERROR HANDLING ORCHESTRATOR
# ============================================================================

class ErrorHandlingOrchestrator:
    """Main orchestrator for error handling features."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.error_handler = ErrorHandlerImpl(config)
        self.circuit_breaker = CircuitBreakerImpl(config)
        self.retry_handler = RetryHandlerImpl(config)
        
        # Recovery strategies
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        
        self.logger.info("Error handling orchestrator initialized")
    
    async def execute_with_protection(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive error protection."""
        try:
            # Execute with circuit breaker if enabled
            if enable_circuit_breaker:
                if enable_retry:
                    # Use retry handler with circuit breaker
                    return await self.retry_handler.execute_with_retry(
                        lambda: self.circuit_breaker.call(func, *args, **kwargs)
                    )
                else:
                    # Use only circuit breaker
                    return await self.circuit_breaker.call(func, *args, **kwargs)
            else:
                if enable_retry:
                    # Use only retry handler
                    return await self.retry_handler.execute_with_retry(func, *args, **kwargs)
                else:
                    # Execute without protection
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
        except Exception as e:
            # Handle error
            error_context = context or {}
            error_context.update({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "circuit_breaker_enabled": enable_circuit_breaker,
                "retry_enabled": enable_retry
            })
            
            await self.error_handler.handle_error(e, error_context)
            
            # Attempt recovery
            recovery_success = await self._attempt_recovery(e, error_context)
            if recovery_success:
                self.logger.info("Error recovery successful")
            else:
                self.logger.warning("Error recovery failed")
            
            # Re-raise error
            raise
    
    async def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
        self.logger.info(f"Added recovery strategy: {type(strategy).__name__}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get error handling system status."""
        try:
            circuit_breaker_state = await self.circuit_breaker.get_state()
            error_summary = await self.error_handler.get_error_summary()
            
            return {
                "status": "active",
                "circuit_breaker": circuit_breaker_state.dict(),
                "error_summary": error_summary,
                "recovery_strategies": len(self.recovery_strategies),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery using available strategies."""
        for strategy in self.recovery_strategies:
            try:
                if await strategy.attempt_recovery(error, context):
                    return True
            except Exception as e:
                self.logger.error(f"Error in recovery strategy {type(strategy).__name__}: {e}")
        
        return False
    
    async def cleanup(self) -> None:
        """Cleanup error handling resources."""
        try:
            # Clear error records
            self.error_handler.errors.clear()
            self.error_handler.error_counts.clear()
            self.error_handler.severity_counts.clear()
            
            # Reset circuit breaker
            self.circuit_breaker.state = "closed"
            self.circuit_breaker.failure_count = 0
            self.circuit_breaker.failure_count = 0
            
            self.logger.info("Error handling orchestrator cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_error_handling_orchestrator(config: ErrorHandlingConfig) -> ErrorHandlingOrchestrator:
    """Create and configure error handling orchestrator."""
    return ErrorHandlingOrchestrator(config)


def create_error_handler(config: ErrorHandlingConfig) -> ErrorHandler:
    """Create and configure error handler."""
    return ErrorHandlerImpl(config)


def create_circuit_breaker(config: ErrorHandlingConfig) -> CircuitBreaker:
    """Create and configure circuit breaker."""
    return CircuitBreakerImpl(config)


def create_retry_handler(config: ErrorHandlingConfig) -> RetryHandler:
    """Create and configure retry handler."""
    return RetryHandlerImpl(config)


def create_fallback_recovery(fallback_func: Optional[Callable] = None) -> ErrorRecoveryStrategy:
    """Create fallback recovery strategy."""
    return FallbackRecoveryStrategy(fallback_func)


def create_graceful_degradation(degraded_services: Dict[str, Callable]) -> ErrorRecoveryStrategy:
    """Create graceful degradation recovery strategy."""
    return GracefulDegradationStrategy(degraded_services)


# ============================================================================
# DECORATORS FOR ERROR HANDLING
# ============================================================================

def with_error_handling(
    orchestrator: Optional[ErrorHandlingOrchestrator] = None,
    enable_circuit_breaker: bool = True,
    enable_retry: bool = True
):
    """Decorator to add error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if orchestrator:
                return await orchestrator.execute_with_protection(
                    func, *args,
                    enable_circuit_breaker=enable_circuit_breaker,
                    enable_retry=enable_retry,
                    **kwargs
                )
            else:
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if orchestrator:
                # For sync functions, we need to handle differently
                # This is a simplified approach
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log error but can't use async orchestrator
                    return None
            else:
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_circuit_breaker(
    orchestrator: Optional[ErrorHandlingOrchestrator] = None
):
    """Decorator to add circuit breaker protection to functions."""
    return with_error_handling(
        orchestrator=orchestrator,
        enable_circuit_breaker=True,
        enable_retry=False
    )


def with_retry(
    orchestrator: Optional[ErrorHandlingOrchestrator] = None
):
    """Decorator to add retry logic to functions."""
    return with_error_handling(
        orchestrator=orchestrator,
        enable_circuit_breaker=False,
        enable_retry=True
    )
