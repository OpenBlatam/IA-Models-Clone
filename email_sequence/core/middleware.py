"""
Custom Middleware for Email Sequence System

This module provides custom middleware for authentication, rate limiting,
caching, and performance monitoring.
"""

import time
import json
import logging
from typing import Callable, Optional, Dict, Any
from uuid import uuid4

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .config import get_settings
from .exceptions import RateLimitError, AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request and response"""
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting"""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check authentication for protected routes"""
        # Skip authentication for public routes
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)
        
        # Check for API key or JWT token
        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        if not auth_header and not api_key:
            # For now, allow requests without authentication
            # In production, you would enforce authentication
            pass
        
        response = await call_next(request)
        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """Caching middleware for GET requests"""
    
    def __init__(self, app: ASGIApp, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply caching for GET requests"""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.url.path}?{request.url.query}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if current_time - cached_data["timestamp"] < self.cache_ttl:
                logger.debug(f"Cache hit for: {cache_key}")
                return JSONResponse(
                    content=cached_data["content"],
                    headers=cached_data["headers"]
                )
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                content = json.loads(response.body.decode())
                self.cache[cache_key] = {
                    "content": content,
                    "headers": dict(response.headers),
                    "timestamp": current_time
                }
                logger.debug(f"Cached response for: {cache_key}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Don't cache non-JSON responses
                pass
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        start_time = time.time()
        
        # Add performance headers
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 1.0:  # Log requests taking more than 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.4f}s"
            )
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"Request [{request_id}]: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'} "
            f"User-Agent: {request.headers.get('user-agent', 'unknown')}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response [{request_id}]: {response.status_code} "
                f"in {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error [{request_id}]: {str(e)} in {process_time:.4f}s",
                exc_info=True
            )
            raise


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """Error tracking middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track errors and exceptions"""
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions (they're handled elsewhere)
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {str(e)}",
                exc_info=True
            )
            
            # In production, you might want to send this to an error tracking service
            # like Sentry
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with more control"""
    
    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = True
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS"""
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
        
        # Add CORS headers
        if origin in self.allow_origins or "*" in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


def setup_custom_middleware(app) -> None:
    """Setup all custom middleware"""
    # Add middleware in reverse order (last added is first executed)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Error tracking
    app.add_middleware(ErrorTrackingMiddleware)
    
    # Enhanced logging
    app.add_middleware(LoggingMiddleware)
    
    # Performance monitoring
    app.add_middleware(PerformanceMiddleware)
    
    # Caching (for GET requests)
    app.add_middleware(CacheMiddleware, cache_ttl=settings.cache_ttl_seconds)
    
    # Authentication
    app.add_middleware(AuthenticationMiddleware)
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute
    )
    
    # Request ID
    app.add_middleware(RequestIDMiddleware)






























