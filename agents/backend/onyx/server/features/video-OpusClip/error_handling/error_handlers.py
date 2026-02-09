#!/usr/bin/env python3
"""
Error Handlers for Video-OpusClip
Handles various types of errors and provides appropriate responses
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
from functools import wraps
import asyncio
from contextlib import contextmanager

from .custom_exceptions import (
    VideoOpusClipException, ValidationError, SecurityError, ScanningError,
    EnumerationError, AttackError, DatabaseError, NetworkError, FileSystemError,
    ConfigurationError, AuthenticationError, AuthorizationError, RateLimitError
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_error_logging(log_level: str = "ERROR", log_file: Optional[str] = None) -> logging.Logger:
    """Setup error logging configuration"""
    logger = logging.getLogger("video_opusclip_errors")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# ERROR RESPONSE GENERATORS
# ============================================================================

def create_error_response(
    error: Exception,
    include_traceback: bool = False,
    include_details: bool = True
) -> Dict[str, Any]:
    """
    Create a standardized error response
    
    Args:
        error: The exception that occurred
        include_traceback: Whether to include stack trace
        include_details: Whether to include error details
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": error.__class__.__name__,
        "message": str(error)
    }
    
    # Handle VideoOpusClip custom exceptions
    if isinstance(error, VideoOpusClipException):
        response.update(error.to_dict())
    else:
        response["error_code"] = "UNKNOWN_ERROR"
        response["details"] = {}
    
    # Add traceback if requested
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    # Add additional details for non-custom exceptions
    if not isinstance(error, VideoOpusClipException) and include_details:
        response["details"] = {
            "exception_type": error.__class__.__name__,
            "module": error.__class__.__module__,
            "args": list(error.args)
        }
    
    return response


def create_user_friendly_error_response(
    error: Exception,
    user_level: str = "user"
) -> Dict[str, Any]:
    """
    Create a user-friendly error response
    
    Args:
        error: The exception that occurred
        user_level: User level ("user", "admin", "developer")
        
    Returns:
        User-friendly error response dictionary
    """
    base_response = create_error_response(error, include_traceback=False, include_details=False)
    
    # Define user-friendly messages
    friendly_messages = {
        ValidationError: "The provided data is invalid. Please check your input and try again.",
        AuthenticationError: "Authentication failed. Please check your credentials and try again.",
        AuthorizationError: "You don't have permission to perform this action.",
        RateLimitError: "Too many requests. Please wait a moment and try again.",
        ConnectionError: "Unable to connect to the database. Please try again later.",
        NetworkError: "Network connection failed. Please check your internet connection.",
        FileNotFoundError: "The requested file was not found.",
        FilePermissionError: "You don't have permission to access this file.",
        ConfigurationError: "System configuration error. Please contact support."
    }
    
    # Get user-friendly message
    friendly_message = friendly_messages.get(type(error), "An unexpected error occurred. Please try again.")
    
    response = {
        "success": False,
        "message": friendly_message,
        "timestamp": base_response["timestamp"],
        "error_code": base_response.get("error_code", "GENERAL_ERROR")
    }
    
    # Add technical details for admin/developer users
    if user_level in ["admin", "developer"]:
        response["technical_details"] = {
            "error_type": error.__class__.__name__,
            "original_message": str(error)
        }
        
        if user_level == "developer":
            response["technical_details"]["traceback"] = traceback.format_exc()
            response["technical_details"]["details"] = base_response.get("details", {})
    
    return response


# ============================================================================
# ERROR HANDLER DECORATORS
# ============================================================================

def handle_errors(
    logger: Optional[logging.Logger] = None,
    reraise: bool = False,
    include_traceback: bool = True
) -> Callable:
    """
    Decorator to handle errors in functions
    
    Args:
        logger: Logger instance for error logging
        reraise: Whether to reraise the exception
        include_traceback: Whether to include traceback in logs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                    else:
                        logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                else:
                    return create_error_response(e, include_traceback=include_traceback)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if logger:
                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                    else:
                        logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                else:
                    return create_error_response(e, include_traceback=include_traceback)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def handle_specific_errors(
    error_types: Dict[type, Callable],
    default_handler: Optional[Callable] = None,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator to handle specific error types with custom handlers
    
    Args:
        error_types: Dictionary mapping exception types to handler functions
        default_handler: Default handler for unhandled exceptions
        logger: Logger instance for error logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Find specific handler
                for error_type, handler in error_types.items():
                    if isinstance(e, error_type):
                        return handler(e, *args, **kwargs)
                
                # Use default handler if provided
                if default_handler:
                    return default_handler(e, *args, **kwargs)
                
                # Log and reraise if no handler found
                if logger:
                    logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Find specific handler
                for error_type, handler in error_types.items():
                    if isinstance(e, error_type):
                        return await handler(e, *args, **kwargs) if asyncio.iscoroutinefunction(handler) else handler(e, *args, **kwargs)
                
                # Use default handler if provided
                if default_handler:
                    return await default_handler(e, *args, **kwargs) if asyncio.iscoroutinefunction(default_handler) else default_handler(e, *args, **kwargs)
                
                # Log and reraise if no handler found
                if logger:
                    logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def error_context(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    error_handler: Optional[Callable] = None
):
    """
    Context manager for error handling
    
    Args:
        operation_name: Name of the operation being performed
        logger: Logger instance for error logging
        reraise: Whether to reraise the exception
        error_handler: Custom error handler function
    """
    try:
        yield
    except Exception as e:
        if logger:
            logger.error(f"Error in {operation_name}: {e}", exc_info=True)
        
        if error_handler:
            error_handler(e, operation_name)
        
        if reraise:
            raise


@contextmanager
def retry_context(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """
    Context manager for retry logic
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to retry on
        logger: Logger instance for logging retry attempts
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            yield
            return
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                if logger:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                
                asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                if logger:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise last_exception


# ============================================================================
# SPECIFIC ERROR HANDLERS
# ============================================================================

def handle_validation_error(error: ValidationError, *args, **kwargs) -> Dict[str, Any]:
    """Handle validation errors"""
    return {
        "success": False,
        "error_code": "VALIDATION_ERROR",
        "message": "Input validation failed",
        "details": {
            "field": error.field,
            "value": str(error.value) if error.value is not None else None,
            "expected_type": error.expected_type,
            "validation_rules": getattr(error, 'validation_rules', [])
        }
    }


def handle_security_error(error: SecurityError, *args, **kwargs) -> Dict[str, Any]:
    """Handle security errors"""
    return {
        "success": False,
        "error_code": "SECURITY_ERROR",
        "message": "Security violation detected",
        "details": {
            "security_level": error.security_level,
            "threat_type": error.threat_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


def handle_database_error(error: DatabaseError, *args, **kwargs) -> Dict[str, Any]:
    """Handle database errors"""
    return {
        "success": False,
        "error_code": "DATABASE_ERROR",
        "message": "Database operation failed",
        "details": {
            "operation": error.operation,
            "table": error.table,
            "query": error.query
        }
    }


def handle_network_error(error: NetworkError, *args, **kwargs) -> Dict[str, Any]:
    """Handle network errors"""
    return {
        "success": False,
        "error_code": "NETWORK_ERROR",
        "message": "Network operation failed",
        "details": {
            "host": error.host,
            "port": error.port,
            "protocol": error.protocol
        }
    }


def handle_file_system_error(error: FileSystemError, *args, **kwargs) -> Dict[str, Any]:
    """Handle file system errors"""
    return {
        "success": False,
        "error_code": "FILE_SYSTEM_ERROR",
        "message": "File system operation failed",
        "details": {
            "file_path": error.file_path,
            "operation": error.operation
        }
    }


# ============================================================================
# ERROR RECOVERY STRATEGIES
# ============================================================================

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
    
    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the given error"""
        raise NotImplementedError
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy for transient errors"""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: tuple = (NetworkError, ConnectionError, TimeoutError),
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, self.retryable_exceptions)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        operation = context.get('operation')
        current_retry = context.get('retry_count', 0)
        
        if current_retry >= self.max_retries:
            if self.logger:
                self.logger.error(f"Max retries ({self.max_retries}) exceeded for {operation}")
            raise error
        
        wait_time = self.delay * (self.backoff_factor ** current_retry)
        
        if self.logger:
            self.logger.warning(f"Retrying {operation} in {wait_time}s (attempt {current_retry + 1}/{self.max_retries})")
        
        await asyncio.sleep(wait_time)
        return {"retry_count": current_retry + 1}


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback strategy for critical errors"""
    
    def __init__(
        self,
        fallback_values: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.fallback_values = fallback_values
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, (ConfigurationError, FileNotFoundError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        operation = context.get('operation')
        fallback_value = self.fallback_values.get(operation)
        
        if self.logger:
            self.logger.warning(f"Using fallback value for {operation}: {fallback_value}")
        
        return fallback_value


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """Circuit breaker strategy for failing operations"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, (DatabaseError, NetworkError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        operation = context.get('operation')
        
        if self.state == "OPEN":
            if datetime.utcnow().timestamp() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                if self.logger:
                    self.logger.info(f"Circuit breaker for {operation} moved to HALF_OPEN")
            else:
                if self.logger:
                    self.logger.warning(f"Circuit breaker for {operation} is OPEN, rejecting request")
                raise error
        
        self.failure_count += 1
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = datetime.utcnow().timestamp()
            if self.logger:
                self.logger.error(f"Circuit breaker for {operation} opened after {self.failure_count} failures")
        
        raise error


# ============================================================================
# ERROR MONITORING
# ============================================================================

class ErrorMonitor:
    """Monitor and track errors for analysis"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.error_counts = {}
        self.error_history = []
        self.max_history_size = 1000
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error occurrence"""
        error_type = error.__class__.__name__
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": str(error),
            "context": context or {}
        }
        
        self.error_history.append(error_record)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        if self.logger:
            self.logger.error(f"Error recorded: {error_type} - {error}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for reporting"""
        if not self.error_counts:
            return {"message": "No errors recorded"}
        
        most_common_error = max(self.error_counts.items(), key=lambda x: x[1])
        
        return {
            "total_errors": sum(self.error_counts.values()),
            "most_common_error": {
                "type": most_common_error[0],
                "count": most_common_error[1]
            },
            "error_distribution": self.error_counts,
            "recent_activity": len([e for e in self.error_history if 
                                  (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).seconds < 3600])
        }


# ============================================================================
# ERROR HANDLER REGISTRY
# ============================================================================

class ErrorHandlerRegistry:
    """Registry for error handlers and recovery strategies"""
    
    def __init__(self):
        self.handlers = {}
        self.recovery_strategies = []
        self.monitor = ErrorMonitor()
    
    def register_handler(self, error_type: type, handler: Callable):
        """Register an error handler"""
        self.handlers[error_type] = handler
    
    def register_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """Register a recovery strategy"""
        self.recovery_strategies.append(strategy)
    
    def get_handler(self, error_type: type) -> Optional[Callable]:
        """Get handler for error type"""
        return self.handlers.get(error_type)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """Handle an error using registered handlers and strategies"""
        # Record error
        self.monitor.record_error(error, context)
        
        # Try specific handler
        handler = self.get_handler(type(error))
        if handler:
            return handler(error, context or {})
        
        # Try recovery strategies
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error):
                return strategy.recover(error, context or {})
        
        # Default handling
        return create_error_response(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return self.monitor.get_error_summary()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of error handlers
    print("🚨 Error Handlers Example")
    
    # Setup logging
    logger = setup_error_logging("INFO")
    
    # Create error handler registry
    registry = ErrorHandlerRegistry()
    
    # Register handlers
    registry.register_handler(ValidationError, handle_validation_error)
    registry.register_handler(SecurityError, handle_security_error)
    registry.register_handler(DatabaseError, handle_database_error)
    registry.register_handler(NetworkError, handle_network_error)
    registry.register_handler(FileSystemError, handle_file_system_error)
    
    # Register recovery strategies
    registry.register_recovery_strategy(RetryStrategy(logger=logger))
    registry.register_recovery_strategy(FallbackStrategy({
        "config_load": {"timeout": 30, "retries": 3}
    }, logger=logger))
    
    # Example function with error handling
    @handle_errors(logger=logger, reraise=False)
    def example_function():
        raise ValidationError("Invalid input", field="email", value="invalid-email")
    
    # Test error handling
    result = example_function()
    print(f"Error handling result: {result}")
    
    # Test registry
    try:
        raise AuthenticationError("Invalid credentials", auth_method="password")
    except Exception as e:
        result = registry.handle_error(e, {"operation": "user_login"})
        print(f"Registry handling result: {result}")
    
    # Get stats
    stats = registry.get_stats()
    print(f"Error stats: {stats}")
    
    # Test context managers
    with error_context("test_operation", logger=logger, reraise=False):
        raise FileNotFoundError("File not found")
    
    # Test retry context
    async def test_retry():
        with retry_context(max_retries=2, delay=0.1, logger=logger):
            raise NetworkError("Connection failed")
    
    # Run async test
    import asyncio
    try:
        asyncio.run(test_retry())
    except Exception as e:
        print(f"Retry test failed as expected: {e}") 