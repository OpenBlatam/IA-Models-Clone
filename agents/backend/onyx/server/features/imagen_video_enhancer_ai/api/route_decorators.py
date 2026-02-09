"""
Route Decorators
================

Common decorators for API routes.
"""

import functools
import logging
from typing import Callable, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors in route handlers.
    
    Args:
        func: Route handler function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return wrapper


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication.
    
    Args:
        func: Route handler function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Check if API key is present
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        # Validate API key (this would use the auth manager)
        # For now, just check if it exists
        return await func(request, *args, **kwargs)
    
    return wrapper


def rate_limit(max_requests: int = 100, window: int = 60):
    """
    Decorator for rate limiting.
    
    Args:
        max_requests: Maximum requests per window
        window: Time window in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Rate limiting logic would go here
            # For now, just pass through
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator


def validate_request(model_class: type):
    """
    Decorator to validate request body.
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Validation logic would go here
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator


def cache_response(ttl: int = 60):
    """
    Decorator to cache responses.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Caching logic would go here
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_request(func: Callable) -> Callable:
    """
    Decorator to log requests.
    
    Args:
        func: Route handler function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        logger.info(f"Request: {request.method} {request.url.path}")
        result = await func(request, *args, **kwargs)
        logger.info(f"Response: {request.method} {request.url.path} - {type(result).__name__}")
        return result
    
    return wrapper


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure performance.
    
    Args:
        func: Route handler function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    
    return wrapper




