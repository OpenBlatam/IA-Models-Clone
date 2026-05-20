"""
Memoization utilities
Advanced memoization patterns
"""

from typing import Callable, TypeVar, Any, Dict, Tuple
from functools import wraps
import hashlib
import json
from utils.cache import cache_result

T = TypeVar('T')


def memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    """
    cache: Dict[Tuple, Any] = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def memoize_with_key(func: Callable, key_func: Callable) -> Callable:
    """
    Memoization with custom key function
    
    Args:
        func: Function to memoize
        key_func: Function to generate cache key
    
    Returns:
        Memoized function
    """
    cache: Dict[Any, Any] = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = key_func(*args, **kwargs)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def memoize_with_hash(func: Callable) -> Callable:
    """
    Memoization using hash of arguments
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    """
    cache: Dict[str, Any] = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create hash of arguments
        key_data = json.dumps({
            "args": args,
            "kwargs": kwargs
        }, sort_keys=True, default=str)
        
        key = hashlib.md5(key_data.encode()).hexdigest()
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def clear_memoization(func: Callable) -> None:
    """
    Clear memoization cache for function
    
    Args:
        func: Memoized function
    """
    if hasattr(func, '__wrapped__'):
        # Clear cache if it exists
        if hasattr(func, 'cache'):
            func.cache.clear()

