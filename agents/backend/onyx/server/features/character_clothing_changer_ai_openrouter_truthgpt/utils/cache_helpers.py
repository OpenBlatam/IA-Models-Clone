"""
Cache Helpers
=============

Helper utilities for caching operations.
"""

import hashlib
import json
from typing import Any, Callable, Optional
from functools import wraps


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Serialize arguments
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    # Convert to JSON string
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    
    # Hash to create fixed-length key
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_key_prefix(prefix: str) -> Callable:
    """
    Decorator to add prefix to cache keys.
    
    Args:
        prefix: Prefix for cache keys
        
    Usage:
        @cache_key_prefix("user")
        def get_user(id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._cache_prefix = prefix
        return func
    return decorator


def make_cache_key(func_name: str, *args, **kwargs) -> str:
    """
    Make cache key from function name and arguments.
    
    Args:
        func_name: Function name
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    prefix = getattr(func_name, '_cache_prefix', '')
    key = generate_cache_key(*args, **kwargs)
    
    if prefix:
        return f"{prefix}:{func_name}:{key}"
    return f"{func_name}:{key}"

