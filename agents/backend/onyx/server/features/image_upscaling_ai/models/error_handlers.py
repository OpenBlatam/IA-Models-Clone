"""
Error Handlers and Retry Logic
================================

Robust error handling and retry mechanisms for upscaling operations.
"""

import logging
import time
from typing import Callable, Type, Tuple, Any, Optional
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    MEMORY = "memory"
    PROCESSING = "processing"
    IO = "io"
    NETWORK = "network"
    UNKNOWN = "unknown"


class UpscalingError(Exception):
    """Base exception for upscaling errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        retryable: bool = False,
        details: Optional[dict] = None
    ):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.details = details or {}


class ValidationError(UpscalingError):
    """Validation error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, ErrorCategory.VALIDATION, retryable=False, details=details)


class MemoryError(UpscalingError):
    """Memory-related error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, ErrorCategory.MEMORY, retryable=False, details=details)


class ProcessingError(UpscalingError):
    """Processing error."""
    def __init__(self, message: str, retryable: bool = True, details: Optional[dict] = None):
        super().__init__(message, ErrorCategory.PROCESSING, retryable=retryable, details=details)


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator to retry operations on failure.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback on retry (attempt, exception)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_upscaling_error(error: Exception) -> UpscalingError:
    """
    Convert generic exceptions to UpscalingError.
    
    Args:
        error: Exception to handle
        
    Returns:
        UpscalingError instance
    """
    error_str = str(error).lower()
    
    # Classify error
    if "memory" in error_str or "out of memory" in error_str:
        return MemoryError(
            str(error),
            details={"original_error": type(error).__name__}
        )
    elif "validation" in error_str or "invalid" in error_str:
        return ValidationError(
            str(error),
            details={"original_error": type(error).__name__}
        )
    elif "network" in error_str or "connection" in error_str:
        return UpscalingError(
            str(error),
            category=ErrorCategory.NETWORK,
            retryable=True,
            details={"original_error": type(error).__name__}
        )
    else:
        return ProcessingError(
            str(error),
            retryable=True,
            details={"original_error": type(error).__name__}
        )


def safe_execute(
    func: Callable,
    default_return: Any = None,
    error_message: str = "Operation failed",
    log_error: bool = True
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        error_message: Error message prefix
        log_error: Whether to log errors
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.error(f"{error_message}: {e}", exc_info=True)
        return default_return


