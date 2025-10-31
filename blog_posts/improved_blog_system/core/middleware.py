"""
Custom middleware for the application
"""

import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


def add_middleware(app: FastAPI):
    """Add custom middleware to the application."""
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)






























