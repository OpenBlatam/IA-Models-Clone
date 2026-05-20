"""
Middleware utilities
Helper functions for creating middleware
"""

from typing import Callable, Any, Optional
import time
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def create_middleware(
    process_request: Optional[Callable[[Request], None]] = None,
    process_response: Optional[Callable[[Request, Response], Response]] = None
) -> type[BaseHTTPMiddleware]:
    """
    Create middleware class from functions
    
    Args:
        process_request: Function to process request
        process_response: Function to process response
    
    Returns:
        Middleware class
    """
    class CustomMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            # Process request
            if process_request:
                await process_request(request) if asyncio.iscoroutinefunction(process_request) else process_request(request)
            
            # Call next middleware
            response = await call_next(request)
            
            # Process response
            if process_response:
                response = await process_response(request, response) if asyncio.iscoroutinefunction(process_response) else process_response(request, response)
            
            return response
    
    return CustomMiddleware


def request_logger(request: Request) -> None:
    """Log request details"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"{request.method} {request.url.path}")


def response_timer(request: Request, response: Response) -> Response:
    """Add timing header to response"""
    if hasattr(request.state, "start_time"):
        duration = time.time() - request.state.start_time
        response.headers["X-Process-Time"] = str(duration)
    return response

