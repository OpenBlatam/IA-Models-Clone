"""
Useful decorators for the application

This module provides decorators for common functionality including:
- Result caching
- Execution time logging
- Retry logic
- Input validation
"""

from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Awaitable, Any
import time
import asyncio
import logging
import hashlib
import json

from utils.cache import cache_service

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """
    Cache function result decorator
    
    Args:
        ttl: Time to live in seconds (default: 3600)
        key_prefix: Prefix for cache key
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key with hash for complex objects
            try:
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                key_hash = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode()
                ).hexdigest()[:8]
                cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
            except Exception:
                # Fallback to simple key
                cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try cache first
            try:
                cached = await cache_service.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache retrieval failed for {func.__name__}: {e}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            try:
                await cache_service.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Cache storage failed for {func.__name__}: {e}")
            
            return result
        return wrapper
    return decorator


def log_execution_time(
    log_level: str = "debug",
    slow_threshold: float = 1.0
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Log function execution time decorator
    
    Args:
        log_level: Logging level (debug, info, warning)
        slow_threshold: Threshold in seconds to log as warning
        
    Returns:
        Decorated function with execution time logging
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_data = {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "execution_time_ms": round(execution_time * 1000, 2)
                }
                
                if execution_time >= slow_threshold:
                    logger.warning(
                        f"{func.__name__} executed slowly in {execution_time:.4f}s",
                        extra=log_data
                    )
                elif log_level == "info":
                    logger.info(
                        f"{func.__name__} executed in {execution_time:.4f}s",
                        extra=log_data
                    )
                else:
                    logger.debug(
                        f"{func.__name__} executed in {execution_time:.4f}s",
                        extra=log_data
                    )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {execution_time:.4f}s: {e}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "error": str(e)
                    }
                )
                raise
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Retry function on failure"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def validate_input(validator: Callable):
    """Validate function input"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validate args and kwargs
            validator(*args, **kwargs)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

