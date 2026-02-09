"""
Service Decorators for Color Grading AI
========================================

Common decorators for services.
"""

import logging
import time
import functools
from typing import Callable, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def track_performance(operation_name: Optional[str] = None):
    """
    Decorator to track performance metrics.
    
    Args:
        operation_name: Optional operation name
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics if service has metrics_collector
                if args and hasattr(args[0], 'metrics_collector'):
                    args[0].metrics_collector.record_operation(
                        operation,
                        duration,
                        success=True
                    )
                
                logger.debug(f"{operation} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if args and hasattr(args[0], 'metrics_collector'):
                    args[0].metrics_collector.record_operation(
                        operation,
                        duration,
                        success=False
                    )
                
                logger.error(f"{operation} failed after {duration:.2f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if args and hasattr(args[0], 'metrics_collector'):
                    args[0].metrics_collector.record_operation(
                        operation,
                        duration,
                        success=True
                    )
                
                logger.debug(f"{operation} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if args and hasattr(args[0], 'metrics_collector'):
                    args[0].metrics_collector.record_operation(
                        operation,
                        duration,
                        success=False
                    )
                
                logger.error(f"{operation} failed after {duration:.2f}s: {e}")
                raise
        
        # Return appropriate wrapper
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_input(validator: Callable):
    """
    Decorator to validate input parameters.
    
    Args:
        validator: Validation function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate arguments
            if not validator(*args, **kwargs):
                raise ValueError("Input validation failed")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(cache_key_func: Optional[Callable] = None, ttl: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        cache_key_func: Function to generate cache key
        ttl: Time to live in seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache manager if available
            cache_manager = None
            if args and hasattr(args[0], 'cache_manager'):
                cache_manager = args[0].cache_manager
            
            if cache_manager:
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{args}:{kwargs}"
                
                # Check cache
                cached = await cache_manager.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute and cache
                result = await func(*args, **kwargs)
                await cache_manager.set(cache_key, result, ttl=ttl)
                return result
            
            # No cache manager, just execute
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar for sync functions
            return func(*args, **kwargs)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def handle_errors(default_return: Any = None, log_error: bool = True):
    """
    Decorator to handle errors gracefully.
    
    Args:
        default_return: Default return value on error
        log_error: Whether to log errors
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




