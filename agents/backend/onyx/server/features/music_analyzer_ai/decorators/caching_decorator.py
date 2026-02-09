"""
Caching Decorator - Cache function results
"""

from typing import Any, Callable, Dict
import hashlib
import json
import functools
import logging

from .decorator import BaseDecorator

logger = logging.getLogger(__name__)


class CachingDecorator(BaseDecorator):
    """
    Decorator that caches function results
    """
    
    def __init__(self, cache: Dict[str, Any] = None, ttl: int = 3600):
        super().__init__("CachingDecorator")
        self.cache = cache or {}
        self.ttl = ttl
    
    def _get_cache_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate cache key"""
        key_data = {
            "func": func.__name__,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def decorate(self, func: Callable) -> Callable:
        """Decorate function with caching"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(func, *args, **kwargs)
            
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return self.cache[cache_key]
            
            result = func(*args, **kwargs)
            self.cache[cache_key] = result
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper


def cache_result(cache: Dict[str, Any] = None):
    """Function decorator for caching"""
    decorator = CachingDecorator(cache=cache)
    return decorator.decorate








