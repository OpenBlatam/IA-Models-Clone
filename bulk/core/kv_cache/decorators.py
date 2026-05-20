"""
Decorators for KV Cache operations.

Provides useful decorators for caching, profiling, and error handling.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def profile_cache_operation(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to profile cache operations.
    
    Example:
        @profile_cache_operation
        def get(self, position: int):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} took {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise
    
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator to retry operations on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries (seconds)
    
    Example:
        @retry_on_failure(max_retries=3, delay=0.1)
        def put(self, position: int, key, value):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            
            raise last_exception  # type: ignore
        
        return wrapper
    
    return decorator


def validate_inputs(
    validate_key: bool = True,
    validate_value: bool = True,
    validate_position: bool = True
):
    """
    Decorator to validate inputs for cache operations.
    
    Args:
        validate_key: Whether to validate key tensor
        validate_value: Whether to validate value tensor
        validate_position: Whether to validate position
        
    Example:
        @validate_inputs(validate_key=True, validate_value=True)
        def put(self, position: int, key, value):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            from kv_cache.validators import CacheValidator
            
            # Validate position if it's the first argument after self
            if validate_position and len(args) > 1:
                position = args[1]
                is_valid, error = CacheValidator.validate_position(position)
                if not is_valid:
                    raise ValueError(f"Invalid position: {error}")
            
            # Validate key/value if present
            if validate_key or validate_value:
                key_arg = kwargs.get("key") or (args[2] if len(args) > 2 else None)
                value_arg = kwargs.get("value") or (args[3] if len(args) > 3 else None)
                
                if key_arg is not None and value_arg is not None:
                    is_valid, error = CacheValidator.validate_tensors(key_arg, value_arg)
                    if not is_valid:
                        raise ValueError(f"Invalid tensors: {error}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def cache_result(ttl: float | None = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (None = infinite)
        
    Example:
        @cache_result(ttl=60.0)
        def expensive_computation(self):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: dict[tuple, tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_key = (args, tuple(sorted(kwargs.items())))
            
            if cache_key in cache:
                result, cached_time = cache[cache_key]
                if ttl is None or (time.time() - cached_time) < ttl:
                    return result
                else:
                    del cache[cache_key]
            
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            return result
        
        return wrapper
    
    return decorator


def synchronized(lock_attr: str = "_lock"):
    """
    Decorator to synchronize method calls using a lock.
    
    Args:
        lock_attr: Name of the lock attribute on the instance
        
    Example:
        @synchronized(lock_attr="_lock")
        def put(self, position: int, key, value):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get lock from instance (first arg is typically self)
            if args:
                instance = args[0]
                lock = getattr(instance, lock_attr, None)
                if lock is not None:
                    with lock:
                        return func(*args, **kwargs)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator



