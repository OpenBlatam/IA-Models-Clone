"""
Logging helper functions for consistent logging patterns.

This module provides utilities for structured logging with context,
performance tracking, and error logging.
"""

import logging
import time
from typing import Any, Optional, Dict, Callable
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


def log_function_call(
    func_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    level: int = logging.INFO
) -> Callable:
    """
    Decorator to log function calls with optional arguments and results.
    
    Args:
        func_name: Custom function name for logging (uses __name__ if None)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        level: Logging level (default: INFO)
    
    Returns:
        Decorator function
    
    Example:
        @log_function_call(log_args=True, log_result=False)
        async def my_endpoint(request: Request):
            return await use_case.execute(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.log(level, f"Calling {name}")
            
            if log_args:
                logger.debug(f"{name} called with args: {args}, kwargs: {kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                
                if log_result:
                    logger.debug(f"{name} returned: {result}")
                
                logger.log(level, f"{name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.log(level, f"Calling {name}")
            
            if log_args:
                logger.debug(f"{name} called with args: {args}, kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                
                if log_result:
                    logger.debug(f"{name} returned: {result}")
                
                logger.log(level, f"{name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)
                raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_performance(
    operation_name: str,
    start_time: float,
    additional_info: Optional[Dict[str, Any]] = None,
    threshold_seconds: float = 1.0
) -> None:
    """
    Log operation performance with timing information.
    
    Args:
        operation_name: Name of the operation
        start_time: Start time from time.time()
        additional_info: Optional additional information to log
        threshold_seconds: Log as warning if exceeds this threshold
    
    Example:
        start = time.time()
        result = await expensive_operation()
        log_performance("expensive_operation", start, {"items": len(result)})
    """
    duration = time.time() - start_time
    log_level = logging.WARNING if duration > threshold_seconds else logging.INFO
    
    message = f"{operation_name} took {duration:.3f}s"
    if additional_info:
        info_str = ", ".join(f"{k}={v}" for k, v in additional_info.items())
        message += f" ({info_str})"
    
    logger.log(log_level, message)


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    operation: Optional[str] = None,
    level: int = logging.ERROR
) -> None:
    """
    Log error with additional context information.
    
    Args:
        error: Exception that occurred
        context: Dictionary with context information
        operation: Optional operation name
        level: Logging level (default: ERROR)
    
    Example:
        try:
            result = await use_case.execute(...)
        except Exception as e:
            log_error_with_context(
                e,
                {"track_id": track_id, "user_id": user_id},
                operation="analyze_track"
            )
            raise
    """
    operation_str = f" in {operation}" if operation else ""
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    
    logger.log(
        level,
        f"Error{operation_str}: {str(error)} | Context: {context_str}",
        exc_info=True
    )


def log_service_call(
    service_name: str,
    method_name: str,
    success: bool = True,
    duration: Optional[float] = None,
    error: Optional[Exception] = None
) -> None:
    """
    Log service method calls with consistent format.
    
    Args:
        service_name: Name of the service
        method_name: Name of the method called
        success: Whether the call was successful
        duration: Optional duration in seconds
        error: Optional error that occurred
    
    Example:
        start = time.time()
        try:
            result = spotify_service.get_track(track_id)
            log_service_call("spotify_service", "get_track", True, time.time() - start)
        except Exception as e:
            log_service_call("spotify_service", "get_track", False, time.time() - start, e)
    """
    status = "success" if success else "failed"
    duration_str = f" ({duration:.3f}s)" if duration else ""
    error_str = f" - {str(error)}" if error else ""
    
    level = logging.ERROR if not success else logging.INFO
    logger.log(
        level,
        f"Service call: {service_name}.{method_name} - {status}{duration_str}{error_str}"
    )


def create_logger_context(
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized logging context dictionary.
    
    Args:
        **kwargs: Context key-value pairs
    
    Returns:
        Dictionary with timestamp and provided context
    
    Example:
        context = create_logger_context(
            track_id=track_id,
            user_id=user_id,
            operation="analyze"
        )
        logger.info("Operation started", extra=context)
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }








