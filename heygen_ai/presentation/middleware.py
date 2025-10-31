from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import time
import uuid
from typing import Callable, Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
            from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Middleware Setup

Configures and sets up middleware for the FastAPI application.
"""


logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_seconds=round(duration, 3)
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed", 
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        
    """__init__ function."""
super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits and process request."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


def setup_middleware(app: FastAPI, settings) -> None:
    """
    Setup middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    logger.info("Setting up middleware")
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Rate limiting middleware (only in production)
    if settings.is_production:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_requests
        )
    
    # Trusted host middleware (only in production)
    if settings.is_production:
        allowed_hosts = ["*"]  # Configure with actual hosts in production
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    logger.info("Middleware setup completed")


# Dependency to get request ID
async def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown") 