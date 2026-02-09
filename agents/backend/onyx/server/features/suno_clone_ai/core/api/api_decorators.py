"""
API Decorators

Decorators for API endpoints.
"""

import logging
import functools
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


def api_endpoint(
    path: str,
    methods: list = None
):
    """
    API endpoint decorator.
    
    Args:
        path: Endpoint path
        methods: HTTP methods
        
    Returns:
        Decorator function
    """
    if methods is None:
        methods = ['GET', 'POST']
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"API endpoint called: {path}")
            return func(*args, **kwargs)
        
        wrapper.path = path
        wrapper.methods = methods
        return wrapper
    
    return decorator


def require_auth(
    auth_func: Optional[Callable] = None
):
    """
    Require authentication decorator.
    
    Args:
        auth_func: Authentication function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(request: Dict[str, Any], *args, **kwargs):
            # Check authentication
            if auth_func:
                if not auth_func(request):
                    return {'success': False, 'error': 'Unauthorized'}
            elif 'token' not in request.get('headers', {}):
                return {'success': False, 'error': 'Unauthorized'}
            
            return func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator


def rate_limited(
    max_requests: int = 100,
    time_window: float = 60.0
):
    """
    Rate limit decorator for API endpoints.
    
    Args:
        max_requests: Maximum requests
        time_window: Time window in seconds
        
    Returns:
        Decorator function
    """
    from core.rate_limit import RateLimiter
    
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(request: Dict[str, Any], *args, **kwargs):
            identifier = request.get('user_id', request.get('ip', 'default'))
            
            if not limiter.is_allowed(identifier):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'retry_after': time_window
                }
            
            return func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator



