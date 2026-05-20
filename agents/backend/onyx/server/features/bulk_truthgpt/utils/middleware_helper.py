"""
Middleware Helper
=================

Advanced middleware utilities for request/response processing.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

def timing_middleware(func: Callable) -> Callable:
    """Middleware to track execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

def error_handler_middleware(func: Callable) -> Callable:
    """Middleware for error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

def rate_limit_middleware(rate: float = 10.0):
    """Middleware for rate limiting."""
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        min_interval = 1.0 / rate
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_called[0]
            
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                await asyncio.sleep(wait_time)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def cache_middleware(ttl: float = 60.0):
    """Middleware for caching results."""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            import hashlib
            import json
            key_data = {
                "args": str(args),
                "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
            }
            cache_key = hashlib.md5(json.dumps(key_data).encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_value, cached_time = cache[cache_key]
                if time.time() - cached_time < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = (result, time.time())
            
            return result
        return wrapper
    return decorator

def retry_middleware(max_attempts: int = 3, delay: float = 1.0):
    """Middleware for automatic retry."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait_time = delay * (2 ** (attempt - 1))
                        logger.warning(f"Attempt {attempt}/{max_attempts} failed, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def logging_middleware(func: Callable) -> Callable:
    """Middleware for request/response logging."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    return wrapper
















