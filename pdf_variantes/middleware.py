"""
Middleware
==========

Custom middleware for PDF variantes feature.
"""

import time
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .exceptions import RateLimitError
from .dependencies import get_config

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Extract request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        request_id = request.headers.get("x-request-id", "")
        
        # Log request
        logger.log(
            self.log_level,
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} - {user_agent} - ID: {request_id}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.log(
                self.log_level,
                f"Response: {response.status_code} "
                f"in {process_time:.3f}s - ID: {request_id}"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error processing request: {e} "
                f"in {process_time:.3f}s - ID: {request_id}"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Clean up old request records."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            cutoff_time = current_time - 60  # Remove requests older than 1 minute
            
            for client_ip in list(self.requests.keys()):
                # Remove old requests
                while (self.requests[client_ip] and 
                       self.requests[client_ip][0] < cutoff_time):
                    self.requests[client_ip].popleft()
                
                # Remove empty entries
                if not self.requests[client_ip]:
                    del self.requests[client_ip]
            
            self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        
        # Clean up old requests
        self._cleanup_old_requests()
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Check if rate limit exceeded
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time <= 60
        ]
        
        return len(recent_requests) > self.requests_per_minute
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/metrics", "/docs", "/openapi.json"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Handle errors globally."""
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": request.headers.get("x-request-id", "")
                }
            )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware for adding request IDs."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Add request ID if not present."""
        request_id = request.headers.get("x-request-id")
        
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"took {process_time:.3f}s (threshold: {self.slow_request_threshold}s)"
            )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching."""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        return f"{request.method}:{request.url.path}:{request.url.query}"
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable."""
        # Only cache GET requests
        if request.method != "GET":
            return False
        
        # Skip certain paths
        skip_paths = ["/health", "/metrics", "/docs"]
        if request.url.path in skip_paths:
            return False
        
        return True
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Handle caching logic."""
        if not self._is_cacheable(request):
            return await call_next(request)
        
        cache_key = self._get_cache_key(request)
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if current_time - cached_data["timestamp"] < self.cache_ttl:
                # Return cached response
                response = JSONResponse(content=cached_data["content"])
                response.headers["X-Cache"] = "HIT"
                response.headers["X-Cache-TTL"] = str(
                    self.cache_ttl - (current_time - cached_data["timestamp"])
                )
                return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                content = response.body.decode() if hasattr(response, 'body') else ""
                self.cache[cache_key] = {
                    "content": content,
                    "timestamp": current_time
                }
                response.headers["X-Cache"] = "MISS"
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        
        return response


def setup_cors_middleware(app, config):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def setup_middleware(app, config):
    """Setup all middleware."""
    # Add middleware in reverse order (last added is first executed)
    
    # Error handling (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Performance monitoring
    app.add_middleware(PerformanceMiddleware, slow_request_threshold=1.0)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, requests_per_minute=config.api.rate_limit_per_minute)
    
    # Caching
    app.add_middleware(CachingMiddleware, cache_ttl=300)
    
    # Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # Logging (innermost)
    app.add_middleware(LoggingMiddleware, log_level=config.logging_level)
    
    # CORS
    setup_cors_middleware(app, config)
