"""
Helper Decorators

Useful decorators for common tasks.
"""

import logging
import time
import functools
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """
    Timer decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    
    return wrapper


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator to cache function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Wrapped function
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def singleton(cls: type) -> type:
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


def deprecated(
    reason: Optional[str] = None,
    version: Optional[str] = None
) -> Callable:
    """
    Deprecated decorator to mark functions as deprecated.
    
    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if version:
                message += f" (since version {version})"
            if reason:
                message += f": {reason}"
            
            logger.warning(message)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator



