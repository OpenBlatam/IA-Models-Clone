"""
Useful Decorators

Common decorators for the application.
"""

import functools
import time
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def singleton(cls):
    """
    Singleton decorator for classes.
    
    Args:
        cls: Class to make singleton
    
    Returns:
        Singleton class
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def deprecated(reason: Optional[str] = None):
    """
    Mark function as deprecated.
    
    Args:
        reason: Optional reason for deprecation
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            logger.warning(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limit decorator.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
    
    Returns:
        Decorator function
    """
    import threading
    
    lock = threading.Lock()
    call_times = []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # Remove old call times
                call_times[:] = [t for t in call_times if now - t < period]
                
                if len(call_times) >= calls:
                    sleep_time = period - (now - call_times[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                        call_times[:] = [t for t in call_times if now - t < period]
                
                call_times.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_args(**validators):
    """
    Validate function arguments.
    
    Args:
        **validators: Dictionary of arg_name -> validator function
    
    Returns:
        Decorator function
    
    Example:
        @validate_args(age=lambda x: x > 0, name=lambda x: len(x) > 0)
        def create_user(name, age):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(f"Invalid argument '{arg_name}': {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Cache function result.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
    
    Returns:
        Decorator function
    """
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            import hashlib
            import json
            key_data = {
                "args": args,
                "kwargs": sorted(kwargs.items())
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                if ttl is None:
                    return cache[cache_key]
                
                # Check expiration
                if time.time() - cache_times[cache_key] < ttl:
                    return cache[cache_key]
                else:
                    # Expired
                    del cache[cache_key]
                    del cache_times[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = result
            cache_times[cache_key] = time.time()
            
            return result
        
        return wrapper
    return decorator



