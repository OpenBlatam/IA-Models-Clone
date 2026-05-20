"""
Safe operation helper functions.

This module provides utilities for executing operations safely,
with error handling that doesn't affect the main flow.
"""

from typing import Any, Callable, Optional, Awaitable
import logging
from functools import wraps

logger = logging.getLogger(__name__)


async def safe_execute(
    operation: Callable[..., Awaitable[Any]],
    *args,
    operation_name: Optional[str] = None,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Execute an async operation safely, catching and logging errors.
    
    Useful for non-critical operations that shouldn't fail the main flow.
    
    Args:
        operation: Async function to execute
        *args: Positional arguments for operation
        operation_name: Optional name for logging
        default_return: Value to return if operation fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation or default_return if it fails
    
    Example:
        await safe_execute(
            history_service.add_analysis,
            track_id,
            analysis,
            operation_name="save_history",
            default_return=None
        )
    """
    name = operation_name or operation.__name__
    
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Safe operation {name} failed: {e}", exc_info=True)
        return default_return


def safe_execute_sync(
    operation: Callable[..., Any],
    *args,
    operation_name: Optional[str] = None,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Execute a sync operation safely, catching and logging errors.
    
    Args:
        operation: Function to execute
        *args: Positional arguments for operation
        operation_name: Optional name for logging
        default_return: Value to return if operation fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation or default_return if it fails
    """
    name = operation_name or operation.__name__
    
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Safe operation {name} failed: {e}", exc_info=True)
        return default_return


async def safe_execute_multiple(
    operations: list[tuple[Callable, tuple, dict, Optional[str]]],
    continue_on_error: bool = True,
    log_errors: bool = True
) -> list[Any]:
    """
    Execute multiple operations safely.
    
    Args:
        operations: List of tuples (operation, args, kwargs, operation_name)
        continue_on_error: Whether to continue if one operation fails
        log_errors: Whether to log errors
    
    Returns:
        List of results (default_return for failed operations)
    
    Example:
        results = await safe_execute_multiple([
            (history_service.add_analysis, (track_id, analysis), {}, "save_history"),
            (analytics_service.track_analysis, (track_id,), {"user_id": user_id}, "track_analytics"),
        ])
    """
    results = []
    
    for operation, args, kwargs, operation_name in operations:
        name = operation_name or operation.__name__
        
        try:
            import inspect
            if inspect.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            results.append(result)
        except Exception as e:
            if log_errors:
                logger.warning(f"Safe operation {name} failed: {e}", exc_info=True)
            
            if continue_on_error:
                results.append(None)
            else:
                raise
    
    return results


def safe_operation(
    operation_name: Optional[str] = None,
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Decorator to make any function execute safely.
    
    Args:
        operation_name: Optional name for logging
        default_return: Value to return if operation fails
        log_errors: Whether to log errors
    
    Returns:
        Decorator function
    
    Example:
        @safe_operation(operation_name="save_to_history", default_return=False)
        async def save_analysis(...):
            # This will never raise, returns False on error
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Safe operation {name} failed: {e}", exc_info=True)
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Safe operation {name} failed: {e}", exc_info=True)
                return default_return
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator








