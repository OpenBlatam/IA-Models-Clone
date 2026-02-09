"""
API Middleware - Custom middleware for the REST API.

This module provides custom middleware functions
for request/response processing.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.
    
    Logs:
    - Request method and path
    - Response status code
    - Request duration
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request and log information."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )
        
        # Add duration header
        response.headers["X-Process-Time"] = str(duration)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for error handling.
    
    Catches exceptions and returns proper error responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request and handle errors."""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
            # FastAPI will handle this, but we log it here
            raise


def setup_middleware(app: ASGIApp):
    """
    Setup all middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)


__all__ = [
    "LoggingMiddleware",
    "ErrorHandlingMiddleware",
    "setup_middleware",
]












