"""
Custom middleware for request/response processing.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - {duration:.3f}s",
            extra={
                "status_code": response.status_code,
                "duration": duration,
                "path": request.url.path
            }
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        return response




class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.request_counts = {
            k: v for k, v in self.request_counts.items()
            if current_time - v["last_reset"] < 60
        }
        
        # Check rate limit
        if client_ip in self.request_counts:
            count_data = self.request_counts[client_ip]
            if count_data["count"] >= self.requests_per_minute:
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": "60"}
                )
            count_data["count"] += 1
        else:
            self.request_counts[client_ip] = {
                "count": 1,
                "last_reset": current_time
            }
        
        response = await call_next(request)
        return response

