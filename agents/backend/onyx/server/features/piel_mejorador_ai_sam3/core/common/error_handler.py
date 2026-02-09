"""
Common Error Handler for Piel Mejorador AI SAM3
===============================================

Unified error handling patterns and decorators.
"""

import logging
import traceback
from typing import Callable, Any, Optional, Dict, Type, Tuple
from functools import wraps
from enum import Enum

from ..error_context import (
    ErrorCategory,
    ErrorContext,
    capture_error_context,
    EnhancedError,
    ProcessingError,
    ValidationError as ContextValidationError,
    NetworkError,
    StorageError,
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Unified error handler."""
    
    @staticmethod
    def handle(
        error: Exception,
        operation_name: str = "operation",
        task_id: Optional[str] = None,
        file_path: Optional[str] = None,
        **metadata
    ) -> ErrorContext:
        """
        Handle error and create context.
        
        Args:
            error: Exception to handle
            operation_name: Name of operation
            task_id: Optional task ID
            file_path: Optional file path
            **metadata: Additional metadata
            
        Returns:
            ErrorContext
        """
        return capture_error_context(
            error,
            task_id=task_id,
            file_path=file_path,
            operation_name=operation_name,
            **metadata
        )
    
    @staticmethod
    def categorize(error: Exception) -> ErrorCategory:
        """
        Categorize error.
        
        Args:
            error: Exception to categorize
            
        Returns:
            ErrorCategory
        """
        if isinstance(error, EnhancedError):
            return error.category
        elif isinstance(error, ValueError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, (IOError, OSError, FileNotFoundError)):
            return ErrorCategory.STORAGE
        else:
            return ErrorCategory.UNKNOWN


def with_error_handling(
    operation_name: Optional[str] = None,
    raise_enhanced: bool = False,
    log_error: bool = True
):
    """
    Decorator for consistent error handling.
    
    Args:
        operation_name: Name of operation for logging
        raise_enhanced: Whether to raise EnhancedError
        log_error: Whether to log errors
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorHandler.handle(
                    e,
                    operation_name=op_name,
                    **kwargs.get("metadata", {})
                )
                
                if log_error:
                    logger.error(
                        f"{op_name} failed: {error_context.message}",
                        exc_info=True,
                        extra={"error_context": error_context.to_dict()}
                    )
                
                if raise_enhanced:
                    category = ErrorHandler.categorize(e)
                    if category == ErrorCategory.VALIDATION:
                        raise ContextValidationError(str(e), error_context.metadata)
                    elif category == ErrorCategory.NETWORK:
                        raise NetworkError(str(e), error_context.metadata)
                    elif category == ErrorCategory.STORAGE:
                        raise StorageError(str(e), error_context.metadata)
                    else:
                        raise ProcessingError(str(e), error_context.metadata)
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorHandler.handle(
                    e,
                    operation_name=op_name,
                    **kwargs.get("metadata", {})
                )
                
                if log_error:
                    logger.error(
                        f"{op_name} failed: {error_context.message}",
                        exc_info=True,
                        extra={"error_context": error_context.to_dict()}
                    )
                
                if raise_enhanced:
                    category = ErrorHandler.categorize(e)
                    if category == ErrorCategory.VALIDATION:
                        raise ContextValidationError(str(e), error_context.metadata)
                    elif category == ErrorCategory.NETWORK:
                        raise NetworkError(str(e), error_context.metadata)
                    elif category == ErrorCategory.STORAGE:
                        raise StorageError(str(e), error_context.metadata)
                    else:
                        raise ProcessingError(str(e), error_context.metadata)
                
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    exponential_backoff: bool = True,
    operation_name: Optional[str] = None
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries
        retryable_exceptions: Tuple of exception types to retry on
        exponential_backoff: Whether to use exponential backoff
        operation_name: Name of operation for logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            import asyncio
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if exponential_backoff:
                            delay = retry_delay * (2 ** attempt)
                        else:
                            delay = retry_delay
                        
                        logger.warning(
                            f"{op_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{op_name} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            import time
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if exponential_backoff:
                            delay = retry_delay * (2 ** attempt)
                        else:
                            delay = retry_delay
                        
                        logger.warning(
                            f"{op_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{op_name} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_error_handling_and_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: Optional[str] = None,
    raise_enhanced: bool = False
):
    """
    Combined decorator for error handling and retry.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries
        retryable_exceptions: Tuple of exception types to retry on
        operation_name: Name of operation for logging
        raise_enhanced: Whether to raise EnhancedError
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        retry_decorated = with_retry(
            max_retries=max_retries,
            retry_delay=retry_delay,
            retryable_exceptions=retryable_exceptions,
            operation_name=operation_name
        )(func)
        
        return with_error_handling(
            operation_name=operation_name,
            raise_enhanced=raise_enhanced
        )(retry_decorated)
    
    return decorator

