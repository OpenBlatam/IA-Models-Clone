"""
Middleware Helper for Document Analyzer
=========================================

Advanced middleware utilities for request/response processing.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Dict, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

def timing_middleware(func: Callable) -> Callable:
    """Middleware to track execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
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
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def error_handler_middleware(func: Callable) -> Callable:
    """Middleware to handle errors gracefully"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def rate_limit_middleware(limiter: Any) -> Callable:
    """Middleware for rate limiting"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await limiter.wait()
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, would need sync limiter
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def cache_middleware(cache_manager: Any, ttl: int = 3600) -> Callable:
    """Middleware for caching"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = cache_manager.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return func
    
    return decorator

def retry_middleware(max_retries: int = 3, backoff: float = 1.0) -> Callable:
    """Middleware for automatic retry"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
        return async_wrapper
    return decorator

def logging_middleware(func: Callable) -> Callable:
    """Middleware for request/response logging"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
















