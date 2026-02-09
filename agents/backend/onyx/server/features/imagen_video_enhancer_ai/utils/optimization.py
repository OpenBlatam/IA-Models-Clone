"""
Optimization Utilities for Imagen Video Enhancer AI
===================================================

Performance optimization utilities.
"""

import logging
import functools
from typing import Callable, Any, Dict
import time

logger = logging.getLogger(__name__)


def cache_result(ttl: int = 3600):
    """
    Cache function results with TTL.
    
    Args:
        ttl: Time to live in seconds
        
    Usage:
        @cache_result(ttl=3600)
        def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, tuple] = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            import hashlib
            import json
            key_data = json.dumps({
                "args": args,
                "kwargs": kwargs
            }, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # Clean old entries
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in cache.items()
                if current_time - ts >= ttl
            ]
            for k in expired_keys:
                del cache[k]
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            import hashlib
            import json
            key_data = json.dumps({
                "args": args,
                "kwargs": kwargs
            }, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # Clean old entries
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in cache.items()
                if current_time - ts >= ttl
            ]
            for k in expired_keys:
                del cache[k]
            
            return result
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper
    
    return decorator


def throttle(calls: int = 10, period: float = 1.0):
    """
    Throttle function calls.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
        
    Usage:
        @throttle(calls=10, period=1.0)
        def api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        import asyncio
        from collections import deque
        
        call_times = deque()
        lock = asyncio.Lock()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with lock:
                now = time.time()
                # Remove old call times
                while call_times and call_times[0] < now - period:
                    call_times.popleft()
                
                # Check if we can make the call
                if len(call_times) >= calls:
                    wait_time = period - (now - call_times[0])
                    if wait_time > 0:
                        logger.debug(f"Throttling {func.__name__}, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                        # Clean again after wait
                        now = time.time()
                        while call_times and call_times[0] < now - period:
                            call_times.popleft()
                
                # Record call time
                call_times.append(time.time())
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import threading
            lock = threading.Lock()
            
            with lock:
                now = time.time()
                # Remove old call times
                while call_times and call_times[0] < now - period:
                    call_times.popleft()
                
                # Check if we can make the call
                if len(call_times) >= calls:
                    wait_time = period - (now - call_times[0])
                    if wait_time > 0:
                        logger.debug(f"Throttling {func.__name__}, waiting {wait_time:.2f}s")
                        time.sleep(wait_time)
                        # Clean again after wait
                        now = time.time()
                        while call_times and call_times[0] < now - period:
                            call_times.popleft()
                
                # Record call time
                call_times.append(time.time())
            
            return func(*args, **kwargs)
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper
    
    return decorator




