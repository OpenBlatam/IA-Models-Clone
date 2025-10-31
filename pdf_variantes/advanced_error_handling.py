"""Advanced error handling with intelligent recovery and monitoring."""

from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from functools import wraps
import asyncio
import time
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import traceback
import sys

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Error context information."""
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class ErrorMetrics:
    """Error metrics tracking."""
    total_errors: int = 0
    errors_by_category: Dict[str, int] = None
    errors_by_severity: Dict[str, int] = None
    errors_by_operation: Dict[str, int] = None
    last_error_time: float = 0
    error_rate: float = 0.0
    
    def __post_init__(self):
        if self.errors_by_category is None:
            self.errors_by_category = defaultdict(int)
        if self.errors_by_severity is None:
            self.errors_by_severity = defaultdict(int)
        if self.errors_by_operation is None:
            self.errors_by_operation = defaultdict(int)


class IntelligentErrorHandler:
    """Intelligent error handler with recovery strategies."""
    
    def __init__(self):
        self._error_metrics = ErrorMetrics()
        self._recovery_strategies = {}
        self._error_patterns = defaultdict(list)
        self._start_time = time.time()
    
    def register_recovery_strategy(
        self,
        error_type: type,
        strategy: Callable[[Exception, ErrorContext], Awaitable[Any]]
    ) -> None:
        """Register recovery strategy for specific error type."""
        self._recovery_strategies[error_type] = strategy
    
    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM
    ) -> Optional[Any]:
        """Handle error with intelligent recovery."""
        # Update metrics
        self._update_metrics(error, context, severity, category)
        
        # Log error
        await self._log_error(error, context, severity, category)
        
        # Try recovery strategy
        recovery_result = await self._try_recovery(error, context)
        if recovery_result is not None:
            logger.info(f"Error recovered: {error.__class__.__name__}")
            return recovery_result
        
        # Check for error patterns
        await self._analyze_error_patterns(error, context)
        
        # Re-raise if no recovery possible
        raise error
    
    def _update_metrics(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> None:
        """Update error metrics."""
        self._error_metrics.total_errors += 1
        self._error_metrics.errors_by_category[category.value] += 1
        self._error_metrics.errors_by_severity[severity.value] += 1
        self._error_metrics.errors_by_operation[context.operation] += 1
        self._error_metrics.last_error_time = time.time()
        
        # Calculate error rate
        uptime = time.time() - self._start_time
        self._error_metrics.error_rate = self._error_metrics.total_errors / uptime
    
    async def _log_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> None:
        """Log error with appropriate level."""
        log_data = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "operation": context.operation,
            "user_id": context.user_id,
            "request_id": context.request_id,
            "severity": severity.value,
            "category": category.value,
            "timestamp": context.timestamp,
            "additional_data": context.additional_data
        }
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {log_data}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {log_data}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {log_data}")
        else:
            logger.info(f"Low severity error: {log_data}")
    
    async def _try_recovery(
        self,
        error: Exception,
        context: ErrorContext
    ) -> Optional[Any]:
        """Try to recover from error."""
        error_type = type(error)
        
        # Check for specific recovery strategy
        if error_type in self._recovery_strategies:
            try:
                return await self._recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Try generic recovery strategies
        return await self._generic_recovery(error, context)
    
    async def _generic_recovery(
        self,
        error: Exception,
        context: ErrorContext
    ) -> Optional[Any]:
        """Generic recovery strategies."""
        # Network errors - retry with backoff
        if "network" in str(error).lower() or "connection" in str(error).lower():
            return await self._retry_with_backoff(context.operation)
        
        # Validation errors - return default value
        if "validation" in str(error).lower() or "invalid" in str(error).lower():
            return self._get_default_value(context.operation)
        
        # Rate limit errors - wait and retry
        if "rate limit" in str(error).lower() or "too many requests" in str(error).lower():
            await asyncio.sleep(1)
            return await self._retry_with_backoff(context.operation)
        
        return None
    
    async def _retry_with_backoff(self, operation: str, max_retries: int = 3) -> Optional[Any]:
        """Retry operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                # This would need to be implemented based on the specific operation
                # For now, return None to indicate no recovery
                return None
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return None
    
    def _get_default_value(self, operation: str) -> Any:
        """Get default value for operation."""
        defaults = {
            "upload": {"success": False, "error": "Upload failed"},
            "process": {"success": False, "error": "Processing failed"},
            "extract": {"topics": [], "error": "Extraction failed"},
            "generate": {"variants": [], "error": "Generation failed"}
        }
        return defaults.get(operation, {"error": "Operation failed"})
    
    async def _analyze_error_patterns(
        self,
        error: Exception,
        context: ErrorContext
    ) -> None:
        """Analyze error patterns for proactive handling."""
        error_key = f"{error.__class__.__name__}:{context.operation}"
        self._error_patterns[error_key].append({
            "timestamp": context.timestamp,
            "user_id": context.user_id,
            "additional_data": context.additional_data
        })
        
        # Keep only recent errors
        cutoff_time = time.time() - 3600  # 1 hour
        self._error_patterns[error_key] = [
            error_data for error_data in self._error_patterns[error_key]
            if error_data["timestamp"] > cutoff_time
        ]
        
        # Check for error spikes
        if len(self._error_patterns[error_key]) > 10:
            logger.warning(f"Error spike detected for {error_key}: {len(self._error_patterns[error_key])} errors in last hour")
    
    def get_error_metrics(self) -> ErrorMetrics:
        """Get error metrics."""
        return self._error_metrics
    
    def get_error_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get error patterns."""
        return dict(self._error_patterns)


class ErrorRecoveryStrategies:
    """Collection of error recovery strategies."""
    
    @staticmethod
    async def network_error_recovery(error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for network errors."""
        if "timeout" in str(error).lower():
            # Retry with longer timeout
            return {"retry": True, "timeout": 30}
        elif "connection refused" in str(error).lower():
            # Wait and retry
            await asyncio.sleep(2)
            return {"retry": True}
        return None
    
    @staticmethod
    async def validation_error_recovery(error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for validation errors."""
        if "required field" in str(error).lower():
            # Return default values
            return {"success": False, "error": "Validation failed", "default": True}
        return None
    
    @staticmethod
    async def rate_limit_recovery(error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for rate limit errors."""
        # Extract retry-after header if available
        retry_after = getattr(error, 'retry_after', 60)
        await asyncio.sleep(min(retry_after, 60))
        return {"retry": True, "retry_after": retry_after}
    
    @staticmethod
    async def database_error_recovery(error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for database errors."""
        if "connection" in str(error).lower():
            # Try to reconnect
            return {"reconnect": True}
        elif "deadlock" in str(error).lower():
            # Retry with delay
            await asyncio.sleep(0.1)
            return {"retry": True}
        return None


def intelligent_error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    operation: str = "unknown"
):
    """Intelligent error handling decorator."""
    error_handler = IntelligentErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                additional_data={"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return await error_handler.handle_error(e, context, severity, category)
        
        return wrapper
    return decorator


def error_recovery(
    recovery_strategy: Callable[[Exception, ErrorContext], Awaitable[Any]]
):
    """Error recovery decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=func.__name__,
                    additional_data={"args": str(args), "kwargs": str(kwargs)}
                )
                
                try:
                    return await recovery_strategy(e, context)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
                    raise e
        
        return wrapper
    return decorator


def error_monitoring(operation_name: str):
    """Error monitoring decorator."""
    error_handler = IntelligentErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation_name,
                additional_data={"function": func.__name__}
            )
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                await error_handler.handle_error(
                    e, context, ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING
                )
                raise e
        
        return wrapper
    return decorator


def graceful_degradation(fallback_value: Any):
    """Graceful degradation decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Operation failed, using fallback: {e}")
                return fallback_value
        
        return wrapper
    return decorator


def error_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Error circuit breaker decorator."""
    from .advanced_performance import CircuitBreaker
    
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call(lambda: func(*args, **kwargs))
            except Exception as e:
                logger.error(f"Circuit breaker triggered for {func.__name__}: {e}")
                raise e
        
        return wrapper
    return decorator


def create_error_handler() -> IntelligentErrorHandler:
    """Create intelligent error handler."""
    handler = IntelligentErrorHandler()
    
    # Register default recovery strategies
    handler.register_recovery_strategy(
        ConnectionError, ErrorRecoveryStrategies.network_error_recovery
    )
    handler.register_recovery_strategy(
        TimeoutError, ErrorRecoveryStrategies.network_error_recovery
    )
    handler.register_recovery_strategy(
        ValueError, ErrorRecoveryStrategies.validation_error_recovery
    )
    
    return handler


def get_error_summary() -> Dict[str, Any]:
    """Get comprehensive error summary."""
    handler = create_error_handler()
    metrics = handler.get_error_metrics()
    patterns = handler.get_error_patterns()
    
    return {
        "metrics": {
            "total_errors": metrics.total_errors,
            "error_rate": metrics.error_rate,
            "errors_by_category": dict(metrics.errors_by_category),
            "errors_by_severity": dict(metrics.errors_by_severity),
            "errors_by_operation": dict(metrics.errors_by_operation),
            "last_error_time": metrics.last_error_time
        },
        "patterns": {
            "error_spikes": {
                key: len(errors) for key, errors in patterns.items()
                if len(errors) > 5
            },
            "recent_errors": {
                key: errors[-5:] for key, errors in patterns.items()
                if errors
            }
        }
    }
