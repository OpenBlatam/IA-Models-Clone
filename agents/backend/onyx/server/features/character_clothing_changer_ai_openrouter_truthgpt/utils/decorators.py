"""
Decorators Utilities
===================

Useful decorators for services and functions.
"""

import logging
import time
import functools
from typing import Callable, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of a function.
    
    Usage:
        @log_execution_time
        async def my_function():
            ...
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
        
    Usage:
        @retry_on_failure(max_retries=3, delay=1.0)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        import asyncio
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            
            raise last_exception
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def cache_result(ttl: float = 3600.0) -> Callable:
    """
    Decorator to cache function results with TTL.
    
    Args:
        ttl: Time to live in seconds
        
    Usage:
        @cache_result(ttl=3600)
        async def my_function(param):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_timestamps = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            import hashlib
            import json
            key_data = {
                "args": str(args),
                "kwargs": json.dumps(kwargs, sort_keys=True)
            }
            cache_key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            now = time.time()
            if cache_key in cache:
                timestamp = cache_timestamps.get(cache_key, 0)
                if now - timestamp < ttl:
                    logger.debug(f"{func.__name__} cache hit")
                    return cache[cache_key]
                else:
                    # Expired, remove from cache
                    del cache[cache_key]
                    del cache_timestamps[cache_key]
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = now
            logger.debug(f"{func.__name__} result cached")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            import hashlib
            import json
            key_data = {
                "args": str(args),
                "kwargs": json.dumps(kwargs, sort_keys=True)
            }
            cache_key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            now = time.time()
            if cache_key in cache:
                timestamp = cache_timestamps.get(cache_key, 0)
                if now - timestamp < ttl:
                    logger.debug(f"{func.__name__} cache hit")
                    return cache[cache_key]
                else:
                    # Expired, remove from cache
                    del cache[cache_key]
                    del cache_timestamps[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = now
            logger.debug(f"{func.__name__} result cached")
            
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_inputs(**validators: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary of parameter name to validator function
        
    Usage:
        @validate_inputs(
            image_url=lambda x: x.startswith('http'),
            num_steps=lambda x: 1 <= x <= 100
        )
        async def my_function(image_url, num_steps):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def rate_limit(calls: int = 100, period: float = 60.0) -> Callable:
    """
    Decorator to rate limit function calls.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
        
    Usage:
        @rate_limit(calls=10, period=60)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        call_history = []
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside period
            call_history[:] = [t for t in call_history if now - t < period]
            
            # Check rate limit
            if len(call_history) >= calls:
                raise Exception(
                    f"Rate limit exceeded: {calls} calls per {period}s"
                )
            
            # Record call
            call_history.append(now)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside period
            call_history[:] = [t for t in call_history if now - t < period]
            
            # Check rate limit
            if len(call_history) >= calls:
                raise Exception(
                    f"Rate limit exceeded: {calls} calls per {period}s"
                )
            
            # Record call
            call_history.append(now)
            
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

