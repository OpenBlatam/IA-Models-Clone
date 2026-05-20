"""
Middleware Helpers
==================

Helper functions for API middleware.
"""

import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to add timing headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add timing headers to response."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response."""
        start_time = time.time()
        
        logger.info(f"Request: {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"- {response.status_code} - {process_time:.3f}s"
        )
        
        return response


class CORSHelper:
    """Helper for CORS configuration."""
    
    @staticmethod
    def get_cors_headers(
        allow_origins: list[str] = None,
        allow_methods: list[str] = None,
        allow_headers: list[str] = None
    ) -> dict[str, str]:
        """
        Get CORS headers.
        
        Args:
            allow_origins: Allowed origins
            allow_methods: Allowed methods
            allow_headers: Allowed headers
            
        Returns:
            Dictionary of CORS headers
        """
        allow_origins = allow_origins or ["*"]
        allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        allow_headers = allow_headers or ["*"]
        
        return {
            "Access-Control-Allow-Origin": ", ".join(allow_origins),
            "Access-Control-Allow-Methods": ", ".join(allow_methods),
            "Access-Control-Allow-Headers": ", ".join(allow_headers),
            "Access-Control-Allow-Credentials": "true"
        }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response




