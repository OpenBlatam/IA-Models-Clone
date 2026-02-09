"""
Logging utilities for professional documents module.

Enhanced logging functions with context and structured logging support.
"""

import logging
import functools
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def log_function_call(func_name: Optional[str] = None, log_args: bool = False):
    """
    Decorator to log function calls with optional argument logging.
    
    Args:
        func_name: Optional custom function name for logging
        log_args: Whether to log function arguments
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.debug(f"Calling {name}")
            
            if log_args:
                logger.debug(f"{name} called with args={args}, kwargs={kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"{name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{name} failed: {str(e)}", exc_info=True)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.debug(f"Calling {name}")
            
            if log_args:
                logger.debug(f"{name} called with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{name} failed: {str(e)}", exc_info=True)
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_document_operation(operation: str, document_id: Optional[str] = None, **context: Any):
    """
    Log a document operation with context.
    
    Args:
        operation: Name of the operation
        document_id: Optional document ID
        **context: Additional context to log
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    if document_id:
        logger.info(f"{operation} - document_id={document_id}, {context_str}")
    else:
        logger.info(f"{operation} - {context_str}")


def log_performance(operation: str, duration: float, **metrics: Any):
    """
    Log performance metrics for an operation.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **metrics: Additional metrics to log
    """
    metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"Performance - {operation}: {duration:.3f}s, {metrics_str}")


def log_error_with_context(
    error: Exception,
    operation: str,
    document_id: Optional[str] = None,
    **context: Any
):
    """
    Log an error with full context.
    
    Args:
        error: The exception that occurred
        operation: Name of the operation that failed
        document_id: Optional document ID
        **context: Additional context
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    log_msg = f"Error in {operation}"
    
    if document_id:
        log_msg += f" - document_id={document_id}"
    
    if context_str:
        log_msg += f", {context_str}"
    
    logger.error(f"{log_msg}: {str(error)}", exc_info=True)






