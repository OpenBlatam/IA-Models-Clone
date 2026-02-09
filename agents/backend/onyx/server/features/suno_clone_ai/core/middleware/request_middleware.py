"""
Request Middleware

Middleware for request/response processing.
"""

import logging
import time
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class RequestMiddleware:
    """Middleware for request processing."""
    
    def __init__(self):
        """Initialize request middleware."""
        self.middlewares: List[Callable] = []
    
    def add(
        self,
        middleware_fn: Callable
    ) -> 'RequestMiddleware':
        """
        Add middleware.
        
        Args:
            middleware_fn: Middleware function
            
        Returns:
            Self for chaining
        """
        self.middlewares.append(middleware_fn)
        return self
    
    def process(
        self,
        request: Dict[str, Any],
        handler: Callable
    ) -> Dict[str, Any]:
        """
        Process request through middleware.
        
        Args:
            request: Request dictionary
            handler: Handler function
            
        Returns:
            Response dictionary
        """
        # Apply middleware
        for middleware in self.middlewares:
            request = middleware(request)
        
        # Call handler
        response = handler(request)
        
        return response


def process_request(
    request: Dict[str, Any],
    handler: Callable,
    *middlewares: Callable
) -> Dict[str, Any]:
    """
    Process request with middleware.
    
    Args:
        request: Request dictionary
        handler: Handler function
        *middlewares: Middleware functions
        
    Returns:
        Response dictionary
    """
    middleware = RequestMiddleware()
    for mw in middlewares:
        middleware.add(mw)
    
    return middleware.process(request, handler)


def process_response(
    response: Dict[str, Any],
    *middlewares: Callable
) -> Dict[str, Any]:
    """
    Process response with middleware.
    
    Args:
        response: Response dictionary
        *middlewares: Middleware functions
        
    Returns:
        Processed response
    """
    processed = response
    for middleware in middlewares:
        processed = middleware(processed)
    
    return processed


# Common middleware functions
def logging_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
    """Log request."""
    logger.info(f"Processing request: {request.get('path', 'unknown')}")
    return request


def timing_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
    """Add timing to request."""
    request['start_time'] = time.time()
    return request


def validation_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate request."""
    # Add validation logic
    return request



