"""
Decorators for common functionality.
"""

from functools import wraps
from typing import Callable, Any
import logging
import time

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of a function.
    
    Usage:
        @log_execution_time
        def my_function():
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{func.__name__} executed in {execution_time:.3f}s",
                extra={"function": func.__name__, "execution_time": execution_time}
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f}s: {e}",
                extra={"function": func.__name__, "execution_time": execution_time, "error": str(e)}
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{func.__name__} executed in {execution_time:.3f}s",
                extra={"function": func.__name__, "execution_time": execution_time}
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f}s: {e}",
                extra={"function": func.__name__, "execution_time": execution_time, "error": str(e)}
            )
            raise
    
    # Return appropriate wrapper based on function type
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors and convert to appropriate exceptions.
    
    Usage:
        @handle_errors
        def my_function():
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def validate_inputs(*validators: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_inputs(validate_user_id, validate_chat_id)
        def my_function(user_id: str, chat_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validators to args
            for i, validator in enumerate(validators):
                if i < len(args):
                    if not validator(args[i]):
                        raise ValueError(f"Invalid input at position {i}")
            return func(*args, **kwargs)
        return wrapper
    return decorator






