"""
Improved Error Handling Module

Enhanced error handling with:
- Early returns and guard clauses
- Structured error responses
- Performance monitoring integration
- Comprehensive logging
- Error recovery strategies
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
import time
import traceback
import structlog
from functools import wraps
from enum import Enum

logger = structlog.get_logger("error_handling")

# =============================================================================
# ERROR CODES AND TYPES
# =============================================================================

class ErrorCode(str, Enum):
    """Standardized error codes."""
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Security errors
    SECURITY_ERROR = "SECURITY_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    MALICIOUS_INPUT = "MALICIOUS_INPUT"
    
    # Processing errors
    PROCESSING_ERROR = "PROCESSING_ERROR"
    VIDEO_PROCESSING_ERROR = "VIDEO_PROCESSING_ERROR"
    BATCH_PROCESSING_ERROR = "BATCH_PROCESSING_ERROR"
    LANGCHAIN_ERROR = "LANGCHAIN_ERROR"
    
    # Resource errors
    RESOURCE_ERROR = "RESOURCE_ERROR"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    DISK_FULL = "DISK_FULL"
    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    YOUTUBE_API_ERROR = "YOUTUBE_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    
    # System errors
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PROCESSING_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = time.time()

class ValidationError(VideoProcessingError):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            details=details
        )
        self.field = field
        self.value = value

class SecurityError(VideoProcessingError):
    """Exception for security-related errors."""
    
    def __init__(
        self,
        message: str,
        threat_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SECURITY_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details
        )
        self.threat_type = threat_type

class ResourceError(VideoProcessingError):
    """Exception for resource-related errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details
        )
        self.resource_type = resource_type

class ExternalServiceError(VideoProcessingError):
    """Exception for external service errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )
        self.service_name = service_name

class ConfigurationError(VideoProcessingError):
    """Exception for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details
        )
        self.config_key = config_key

# =============================================================================
# ERROR RESPONSE MODELS
# =============================================================================

class ErrorResponse:
    """Structured error response."""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        severity: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.request_id = request_id
        self.timestamp = timestamp or time.time()
        self.severity = severity or "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "severity": self.severity,
                "timestamp": self.timestamp,
                "request_id": self.request_id
            }
        }

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized error response."""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        severity=severity
    ).to_dict()

def log_error_with_context(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> None:
    """Log error with structured context."""
    context = context or {}
    
    # Extract error information
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "request_id": request_id,
        "timestamp": time.time()
    }
    
    # Add context information
    error_info.update(context)
    
    # Add stack trace for debugging
    if isinstance(error, VideoProcessingError):
        error_info.update({
            "error_code": error.error_code,
            "severity": error.severity,
            "details": error.details
        })
        
        if error.original_error:
            error_info["original_error"] = {
                "type": type(error.original_error).__name__,
                "message": str(error.original_error)
            }
    
    # Log based on severity
    if isinstance(error, VideoProcessingError):
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **error_info)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **error_info)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **error_info)
        else:
            logger.info("Low severity error occurred", **error_info)
    else:
        logger.error("Unhandled error occurred", **error_info)

# =============================================================================
# ERROR HANDLING DECORATORS
# =============================================================================

def handle_processing_errors(func: Callable) -> Callable:
    """Decorator to handle processing errors with structured logging."""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        request_id = None
        
        # Extract request ID if available
        for arg in args:
            if hasattr(arg, 'state') and hasattr(arg.state, 'request_id'):
                request_id = arg.state.request_id
                break
        
        try:
            return await func(*args, **kwargs)
        
        except ValidationError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except SecurityError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except ResourceError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except ExternalServiceError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except VideoProcessingError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except Exception as e:
            # Convert unknown exceptions to VideoProcessingError
            processing_error = VideoProcessingError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code=ErrorCode.INTERNAL_ERROR,
                severity=ErrorSeverity.HIGH,
                original_error=e
            )
            log_error_with_context(processing_error, {"function": func.__name__}, request_id)
            raise processing_error
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        request_id = None
        
        # Extract request ID if available
        for arg in args:
            if hasattr(arg, 'state') and hasattr(arg.state, 'request_id'):
                request_id = arg.state.request_id
                break
        
        try:
            return func(*args, **kwargs)
        
        except ValidationError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except SecurityError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except ResourceError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except ExternalServiceError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except VideoProcessingError as e:
            log_error_with_context(e, {"function": func.__name__}, request_id)
            raise
        
        except Exception as e:
            # Convert unknown exceptions to VideoProcessingError
            processing_error = VideoProcessingError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code=ErrorCode.INTERNAL_ERROR,
                severity=ErrorSeverity.HIGH,
                original_error=e
            )
            log_error_with_context(processing_error, {"function": func.__name__}, request_id)
            raise processing_error
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def handle_validation_errors(func: Callable) -> Callable:
    """Decorator to handle validation errors specifically."""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(
                "Validation error",
                error=str(e),
                field=getattr(e, 'field', None),
                value=getattr(e, 'value', None),
                function=func.__name__
            )
            raise
        except Exception as e:
            # Convert other exceptions to ValidationError
            validation_error = ValidationError(
                message=f"Validation failed in {func.__name__}: {str(e)}",
                original_error=e
            )
            logger.warning("Validation error", error=str(validation_error), function=func.__name__)
            raise validation_error
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(
                "Validation error",
                error=str(e),
                field=getattr(e, 'field', None),
                value=getattr(e, 'value', None),
                function=func.__name__
            )
            raise
        except Exception as e:
            # Convert other exceptions to ValidationError
            validation_error = ValidationError(
                message=f"Validation failed in {func.__name__}: {str(e)}",
                original_error=e
            )
            logger.warning("Validation error", error=str(validation_error), function=func.__name__)
            raise validation_error
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# =============================================================================
# ERROR RECOVERY STRATEGIES
# =============================================================================

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can recover from the error."""
        return False
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError

class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy for transient errors."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
    
    def can_recover(self, error: Exception) -> bool:
        """Check if error is retryable."""
        retryable_errors = (
            ExternalServiceError,
            ResourceError,
            TimeoutError,
            ConnectionError
        )
        return isinstance(error, retryable_errors)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry the operation."""
        import asyncio
        
        func = context.get('function')
        args = context.get('args', [])
        kwargs = context.get('kwargs', {})
        
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as retry_error:
                if attempt == self.max_retries - 1:
                    raise retry_error
                
                wait_time = self.delay * (self.backoff ** attempt)
                logger.info(
                    f"Retry attempt {attempt + 1} failed, waiting {wait_time}s",
                    error=str(retry_error),
                    attempt=attempt + 1,
                    max_retries=self.max_retries
                )
                await asyncio.sleep(wait_time)

class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback strategy for service failures."""
    
    def __init__(self, fallback_config: Dict[str, Any]):
        self.fallback_config = fallback_config
    
    def can_recover(self, error: Exception) -> bool:
        """Check if fallback is available."""
        return isinstance(error, ExternalServiceError)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use fallback configuration."""
        service_name = getattr(error, 'service_name', 'unknown')
        fallback = self.fallback_config.get(service_name)
        
        if fallback:
            logger.info(
                f"Using fallback for {service_name}",
                fallback=fallback,
                original_error=str(error)
            )
            return fallback
        
        raise error

# =============================================================================
# ERROR MONITORING
# =============================================================================

class ErrorMonitor:
    """Monitor and track errors for analysis."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error for monitoring."""
        error_type = type(error).__name__
        error_code = getattr(error, 'error_code', 'UNKNOWN')
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_counts[f"{error_type}:{error_code}"] = self.error_counts.get(f"{error_type}:{error_code}", 0) + 1
        
        # Record error history
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_code": error_code,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.error_history.append(error_record)
        
        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts,
            "total_errors": len(self.error_history),
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "error_rate": self._calculate_error_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per minute."""
        if not self.error_history:
            return 0.0
        
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        recent_errors = [
            error for error in self.error_history
            if error["timestamp"] > one_minute_ago
        ]
        
        return len(recent_errors) / 60.0  # errors per second

# Global error monitor
error_monitor = ErrorMonitor()

# =============================================================================
# ERROR HANDLER REGISTRY
# =============================================================================

class ErrorHandlerRegistry:
    """Registry for error handlers and recovery strategies."""
    
    def __init__(self):
        self.handlers: Dict[type, Callable] = {}
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
    
    def register_handler(self, exception_type: type, handler: Callable) -> None:
        """Register an error handler for a specific exception type."""
        self.handlers[exception_type] = handler
    
    def register_recovery_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Register an error recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle an error using registered handlers and recovery strategies."""
        # Record error for monitoring
        error_monitor.record_error(error, context)
        
        # Try recovery strategies first
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error):
                try:
                    return strategy.recover(error, context or {})
                except Exception as recovery_error:
                    logger.warning(
                        "Recovery strategy failed",
                        original_error=str(error),
                        recovery_error=str(recovery_error),
                        strategy=type(strategy).__name__
                    )
        
        # Use registered handler
        handler = self.handlers.get(type(error))
        if handler:
            return handler(error, context)
        
        # Default handling
        log_error_with_context(error, context)
        raise error

# Global error handler registry
error_handler_registry = ErrorHandlerRegistry()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Error codes and types
    'ErrorCode',
    'ErrorSeverity',
    
    # Custom exceptions
    'VideoProcessingError',
    'ValidationError',
    'SecurityError',
    'ResourceError',
    'ExternalServiceError',
    'ConfigurationError',
    
    # Error response models
    'ErrorResponse',
    
    # Error handling utilities
    'create_error_response',
    'log_error_with_context',
    
    # Error handling decorators
    'handle_processing_errors',
    'handle_validation_errors',
    
    # Error recovery strategies
    'ErrorRecoveryStrategy',
    'RetryStrategy',
    'FallbackStrategy',
    
    # Error monitoring
    'ErrorMonitor',
    'error_monitor',
    
    # Error handler registry
    'ErrorHandlerRegistry',
    'error_handler_registry'
]