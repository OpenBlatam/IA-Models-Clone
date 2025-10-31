"""
PDF Variantes API - Response Decorators
Decorators for enhancing API responses
"""

import time
import functools
from typing import Callable, Any, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from .responses import create_success_response, create_error_response


def standard_response(
    wrap_data: bool = True,
    add_metadata: bool = True
):
    """
    Decorator to wrap responses in standard format
    
    Args:
        wrap_data: Whether to wrap data in SuccessResponse
        add_metadata: Whether to add request metadata
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs if available
            request: Request = kwargs.get('request') or (
                args[0] if args and isinstance(args[0], Request) else None
            )
            
            request_id = None
            if request:
                request_id = getattr(request.state, 'request_id', None)
            
            try:
                # Execute the function
                result = await func(*args, **kwargs) if callable(func) else func(*args, **kwargs)
                
                # If result is already a response, return it
                if isinstance(result, Response) or isinstance(result, JSONResponse):
                    return result
                
                # Wrap in standard format if requested
                if wrap_data:
                    return create_success_response(
                        data=result,
                        request_id=request_id
                    )
                return result
                
            except Exception as e:
                # Don't catch HTTPException, let it propagate
                from fastapi import HTTPException
                if isinstance(e, HTTPException):
                    raise
                
                # Wrap other exceptions
                return create_error_response(
                    message=str(e),
                    status_code=500,
                    error_type="InternalServerError",
                    request_id=request_id
                )
        
        return wrapper
    return decorator


def cache_response(max_age: int = 300):
    """
    Decorator to add cache headers to responses
    
    Args:
        max_age: Cache max age in seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            response = await func(*args, **kwargs)
            
            # Add cache headers if response is JSONResponse
            if isinstance(response, Response):
                response.headers["Cache-Control"] = f"public, max-age={max_age}"
                response.headers["X-Cache-Status"] = "HIT"
            
            return response
        
        return wrapper
    return decorator


def timing_decorator(func: Callable):
    """Decorator to add timing information to response"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Add timing header if result is Response
            if isinstance(result, Response):
                elapsed = time.time() - start_time
                result.headers["X-Response-Time"] = f"{elapsed:.3f}s"
            
            return result
        except Exception:
            elapsed = time.time() - start_time
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Request failed after {elapsed:.3f}s")
            raise
    
    return wrapper


def validate_request(func: Callable):
    """Decorator to validate request parameters"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Validate pagination if present
        if 'page' in kwargs and 'limit' in kwargs:
            page = kwargs['page']
            limit = kwargs['limit']
            if page < 1:
                raise ValueError("page must be >= 1")
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
        
        return await func(*args, **kwargs)
    
    return wrapper






