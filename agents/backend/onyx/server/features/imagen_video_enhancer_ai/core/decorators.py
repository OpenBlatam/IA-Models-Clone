"""
Common Decorators
================

Shared decorators for the enhancer agent.
"""

import functools
import time
import logging
import asyncio
from typing import Callable, Any, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def async_or_sync(func: Callable) -> Callable:
    """
    Decorator that works for both async and sync functions.
    
    Automatically detects function type and applies appropriate wrapper.
    """
    if asyncio.iscoroutinefunction(func):
        return func
    return func


def measure_time(log_level: str = "debug") -> Callable:
    """
    Decorator to measure and log execution time.
    
    Args:
        log_level: Logging level (debug, info, warning)
        
    Usage:
        @measure_time()
        async def my_function():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                log_func = getattr(logger, log_level.lower(), logger.debug)
                log_func(f"{func.__name__} took {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                log_func = getattr(logger, log_level.lower(), logger.debug)
                log_func(f"{func.__name__} took {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_calls(include_args: bool = False, include_result: bool = False) -> Callable:
    """
    Decorator to log function calls.
    
    Args:
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        
    Usage:
        @log_calls(include_args=True)
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log_msg = f"Calling {func.__name__}"
            if include_args:
                log_msg += f" with args={args}, kwargs={kwargs}"
            logger.debug(log_msg)
            
            try:
                result = await func(*args, **kwargs)
                if include_result:
                    logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log_msg = f"Calling {func.__name__}"
            if include_args:
                log_msg += f" with args={args}, kwargs={kwargs}"
            logger.debug(log_msg)
            
            try:
                result = func(*args, **kwargs)
                if include_result:
                    logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def handle_errors(default_return: Any = None, log_error: bool = True) -> Callable:
    """
    Decorator to handle errors gracefully.
    
    Args:
        default_return: Value to return on error
        log_error: Whether to log errors
        
    Usage:
        @handle_errors(default_return=None)
        async def my_function():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_input(validator: Callable[[Any], bool], error_message: str = "Validation failed") -> Callable:
    """
    Decorator to validate function input.
    
    Args:
        validator: Validation function
        error_message: Error message on validation failure
        
    Usage:
        @validate_input(lambda x: x > 0, "Value must be positive")
        def my_function(value: int):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validate first argument
            if args and not validator(args[0]):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator




