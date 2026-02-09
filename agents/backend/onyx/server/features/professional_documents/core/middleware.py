"""
Middleware utilities for professional documents API.

Request/response interceptors and middleware functions.
"""

import time
import logging
from typing import Callable, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .logging_utils import log_performance

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with timing information."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log timing information."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            log_performance(
                f"{request.method} {request.url.path}",
                duration,
                status_code=response.status_code
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {duration:.3f}s: {str(e)}"
            )
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware (basic implementation)."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._request_counts: dict[str, list[float]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self._request_counts:
            self._request_counts[client_ip] = [
                req_time
                for req_time in self._request_counts[client_ip]
                if current_time - req_time < 60
            ]
        
        # Check rate limit
        if client_ip in self._request_counts:
            request_count = len(self._request_counts[client_ip])
            if request_count >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return Response(
                    content="Rate limit exceeded. Please try again later.",
                    status_code=429,
                    headers={"Retry-After": "60"}
                )
        
        # Record request
        if client_ip not in self._request_counts:
            self._request_counts[client_ip] = []
        self._request_counts[client_ip].append(current_time)
        
        return await call_next(request)


def add_request_id_middleware(app):
    """Add request ID to all requests for tracing."""
    import uuid
    
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Callable):
        """Add unique request ID to request and response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    return app
