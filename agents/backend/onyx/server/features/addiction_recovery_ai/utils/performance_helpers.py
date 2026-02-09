"""
Performance optimization helpers
Utilities for performance monitoring and optimization
"""

from typing import Callable, Any, Dict
from functools import wraps
import time
from utils.metrics import record_request_metric


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
    
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Record metric if possible
        try:
            record_request_metric(
                endpoint=func.__name__,
                method="FUNCTION",
                status_code=200,
                duration=duration
            )
        except Exception:
            pass  # Ignore metric recording errors
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Record metric if possible
        try:
            record_request_metric(
                endpoint=func.__name__,
                method="FUNCTION",
                status_code=200,
                duration=duration
            )
        except Exception:
            pass  # Ignore metric recording errors
        
        return result
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    return sync_wrapper


def batch_process(
    items: list[Any],
    batch_size: int,
    processor: Callable[[list[Any]], Any]
) -> list[Any]:
    """
    Process items in batches
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Function to process each batch
    
    Returns:
        List of processed results
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = processor(batch)
        results.append(result)
    
    return results


def memoize_with_ttl(ttl_seconds: int = 300):
    """
    Memoization decorator with TTL
    
    Args:
        ttl_seconds: Time to live in seconds
    
    Returns:
        Decorator function
    """
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if cached and not expired
            if key in cache:
                cache_time = cache_times.get(key, 0)
                if time.time() - cache_time < ttl_seconds:
                    return cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        return wrapper
    
    return decorator

