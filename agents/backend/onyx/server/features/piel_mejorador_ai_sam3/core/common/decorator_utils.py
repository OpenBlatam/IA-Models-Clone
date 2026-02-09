"""
Decorator Utilities for Piel Mejorador AI SAM3
==============================================

Unified decorator patterns and utilities.
"""

import asyncio
import functools
import logging
from typing import Callable, Any, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class DecoratorUtils:
    """Unified decorator utilities."""
    
    @staticmethod
    def async_sync_wrapper(func: Callable) -> Callable:
        """
        Create wrapper that handles both async and sync functions.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapper function
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return sync_wrapper
    
    @staticmethod
    def create_decorator(
        before: Optional[Callable] = None,
        after: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_finally: Optional[Callable] = None
    ) -> Callable:
        """
        Create a decorator with before/after/error/finally hooks.
        
        Args:
            before: Function to call before wrapped function
            after: Function to call after wrapped function (receives result)
            on_error: Function to call on error (receives exception)
            on_finally: Function to call in finally block
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        if before:
                            if asyncio.iscoroutinefunction(before):
                                await before(*args, **kwargs)
                            else:
                                before(*args, **kwargs)
                        
                        result = await func(*args, **kwargs)
                        
                        if after:
                            if asyncio.iscoroutinefunction(after):
                                await after(result, *args, **kwargs)
                            else:
                                after(result, *args, **kwargs)
                        
                        return result
                    except Exception as e:
                        if on_error:
                            if asyncio.iscoroutinefunction(on_error):
                                await on_error(e, *args, **kwargs)
                            else:
                                on_error(e, *args, **kwargs)
                        raise
                    finally:
                        if on_finally:
                            if asyncio.iscoroutinefunction(on_finally):
                                await on_finally(*args, **kwargs)
                            else:
                                on_finally(*args, **kwargs)
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    try:
                        if before:
                            before(*args, **kwargs)
                        
                        result = func(*args, **kwargs)
                        
                        if after:
                            after(result, *args, **kwargs)
                        
                        return result
                    except Exception as e:
                        if on_error:
                            on_error(e, *args, **kwargs)
                        raise
                    finally:
                        if on_finally:
                            on_finally(*args, **kwargs)
                
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def memoize(
        ttl: Optional[float] = None,
        max_size: Optional[int] = None,
        key_func: Optional[Callable] = None
    ) -> Callable:
        """
        Memoization decorator with optional TTL and size limit.
        
        Args:
            ttl: Time to live in seconds (None = no expiration)
            max_size: Maximum cache size (None = unlimited)
            key_func: Custom key function (default: uses args and kwargs)
            
        Returns:
            Memoization decorator
        """
        import time
        from collections import OrderedDict
        
        cache = OrderedDict()
        cache_times = {}
        
        def decorator(func: Callable) -> Callable:
            def get_key(*args, **kwargs):
                if key_func:
                    return key_func(*args, **kwargs)
                return (args, tuple(sorted(kwargs.items())))
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    key = get_key(*args, **kwargs)
                    now = time.time()
                    
                    # Check cache
                    if key in cache:
                        if ttl is None or (now - cache_times[key]) < ttl:
                            # Move to end (LRU)
                            cache.move_to_end(key)
                            return cache[key]
                        else:
                            # Expired
                            del cache[key]
                            del cache_times[key]
                    
                    # Call function
                    result = await func(*args, **kwargs)
                    
                    # Store in cache
                    cache[key] = result
                    cache_times[key] = now
                    
                    # Enforce max size
                    if max_size and len(cache) > max_size:
                        oldest = next(iter(cache))
                        del cache[oldest]
                        del cache_times[oldest]
                    
                    return result
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    key = get_key(*args, **kwargs)
                    now = time.time()
                    
                    # Check cache
                    if key in cache:
                        if ttl is None or (now - cache_times[key]) < ttl:
                            # Move to end (LRU)
                            cache.move_to_end(key)
                            return cache[key]
                        else:
                            # Expired
                            del cache[key]
                            del cache_times[key]
                    
                    # Call function
                    result = func(*args, **kwargs)
                    
                    # Store in cache
                    cache[key] = result
                    cache_times[key] = now
                    
                    # Enforce max size
                    if max_size and len(cache) > max_size:
                        oldest = next(iter(cache))
                        del cache[oldest]
                        del cache_times[oldest]
                    
                    return result
                
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def rate_limit(
        max_calls: int,
        period: float
    ) -> Callable:
        """
        Rate limiting decorator.
        
        Args:
            max_calls: Maximum number of calls
            period: Time period in seconds
            
        Returns:
            Rate limiting decorator
        """
        import time
        from collections import deque
        
        calls = deque()
        
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    now = time.time()
                    
                    # Remove old calls
                    while calls and calls[0] < now - period:
                        calls.popleft()
                    
                    # Check rate limit
                    if len(calls) >= max_calls:
                        sleep_time = period - (now - calls[0])
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                            # Re-check after sleep
                            now = time.time()
                            while calls and calls[0] < now - period:
                                calls.popleft()
                    
                    # Record call
                    calls.append(now)
                    
                    return await func(*args, **kwargs)
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    now = time.time()
                    
                    # Remove old calls
                    while calls and calls[0] < now - period:
                        calls.popleft()
                    
                    # Check rate limit
                    if len(calls) >= max_calls:
                        sleep_time = period - (now - calls[0])
                        if sleep_time > 0:
                            import time
                            time.sleep(sleep_time)
                            # Re-check after sleep
                            now = time.time()
                            while calls and calls[0] < now - period:
                                calls.popleft()
                    
                    # Record call
                    calls.append(now)
                    
                    return func(*args, **kwargs)
                
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def timeout(
        timeout_seconds: float,
        default: Optional[Any] = None,
        on_timeout: Optional[Callable] = None
    ) -> Callable:
        """
        Timeout decorator.
        
        Args:
            timeout_seconds: Timeout in seconds
            default: Default value to return on timeout
            on_timeout: Function to call on timeout
            
        Returns:
            Timeout decorator
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        return await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        if on_timeout:
                            if asyncio.iscoroutinefunction(on_timeout):
                                await on_timeout(*args, **kwargs)
                            else:
                                on_timeout(*args, **kwargs)
                        return default
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Function {func.__name__} timed out")
                    
                    # Set signal handler
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout_seconds))
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except TimeoutError:
                        if on_timeout:
                            on_timeout(*args, **kwargs)
                        return default
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                
                return sync_wrapper
        
        return decorator


# Convenience functions
def async_sync_wrapper(func: Callable) -> Callable:
    """Create async/sync wrapper."""
    return DecoratorUtils.async_sync_wrapper(func)


def create_decorator(**kwargs) -> Callable:
    """Create decorator with hooks."""
    return DecoratorUtils.create_decorator(**kwargs)


def memoize(**kwargs) -> Callable:
    """Memoization decorator."""
    return DecoratorUtils.memoize(**kwargs)


def rate_limit(max_calls: int, period: float) -> Callable:
    """Rate limiting decorator."""
    return DecoratorUtils.rate_limit(max_calls, period)


def timeout(timeout_seconds: float, **kwargs) -> Callable:
    """Timeout decorator."""
    return DecoratorUtils.timeout(timeout_seconds, **kwargs)




