"""
Advanced middleware for Facebook Posts API
Following FastAPI best practices for middleware implementation
"""

import time
import uuid
import asyncio
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import get_settings

logger = structlog.get_logger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking request timing and performance"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with timing information"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            request_id=request_id,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add timing headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Timestamp"] = str(int(start_time))
            
            # Log request completion
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                request_id=request_id
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time
            
            # Log request error
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
                request_id=request_id,
                exc_info=True
            )
            
            # Create error response
            error_response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "request_id": request_id,
                    "timestamp": time.time()
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )
            
            return error_response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.requests = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        client_id = f"{client_ip}:{hash(user_agent)}"
        
        # Check rate limit
        if not await self._check_rate_limit(client_id):
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                client_ip=client_ip,
                url=str(request.url)
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": self.settings.rate_limit_window,
                    "timestamp": time.time()
                },
                headers={
                    "Retry-After": str(self.settings.rate_limit_window),
                    "X-RateLimit-Limit": str(self.settings.rate_limit_requests),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.settings.rate_limit_window))
        
        return response
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get client's request history
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        client_requests = self.requests[client_id]
        
        # Remove requests outside the window
        cutoff_time = current_time - self.settings.rate_limit_window
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        # Check if under limit
        if len(client_requests) >= self.settings.rate_limit_requests:
            return False
        
        # Add current request
        client_requests.append(current_time)
        return True
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.requests:
            return self.settings.rate_limit_requests
        
        current_time = time.time()
        cutoff_time = current_time - self.settings.rate_limit_window
        client_requests = self.requests[client_id]
        
        # Count requests within window
        recent_requests = [req_time for req_time in client_requests if req_time > cutoff_time]
        return max(0, self.settings.rate_limit_requests - len(recent_requests))
    
    async def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limit entries"""
        cutoff_time = current_time - 3600  # 1 hour
        
        for client_id in list(self.requests.keys()):
            client_requests = self.requests[client_id]
            client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
            
            if not client_requests:
                del self.requests[client_id]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security headers"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Add HSTS header for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add CSP header
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with compression"""
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        
        if "gzip" not in accept_encoding:
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Only compress certain content types
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in ["application/json", "text/", "application/javascript"]):
            return response
        
        # Only compress responses above certain size
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < 1024:  # Less than 1KB
            return response
        
        # Add compression header
        response.headers["Content-Encoding"] = "gzip"
        
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for cache control headers"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with cache control"""
        response = await call_next(request)
        
        # Set cache headers based on endpoint
        if request.url.path.startswith("/api/v1/posts") and request.method == "GET":
            # Cache GET requests for posts
            response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
            response.headers["ETag"] = f'"{hash(str(request.url))}"'
        elif request.url.path in ["/health", "/metrics"]:
            # Short cache for health and metrics
            response.headers["Cache-Control"] = "public, max-age=60"  # 1 minute
        elif request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            # Longer cache for documentation
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour
        else:
            # No cache for other endpoints
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware for limiting request size"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with size limits"""
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.settings.max_request_size:
            logger.warning(
                "Request too large",
                content_length=content_length,
                max_size=self.settings.max_request_size,
                url=str(request.url)
            )
            
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request entity too large",
                    "error_code": "REQUEST_TOO_LARGE",
                    "max_size": self.settings.max_request_size,
                    "timestamp": time.time()
                }
            )
        
        return await call_next(request)


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for limiting concurrent requests"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        self.active_requests = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with concurrency limits"""
        # Check if we can process the request
        if self.active_requests >= self.settings.max_concurrent_requests:
            logger.warning(
                "Too many concurrent requests",
                active_requests=self.active_requests,
                max_concurrent=self.settings.max_concurrent_requests,
                url=str(request.url)
            )
            
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service temporarily unavailable",
                    "error_code": "TOO_MANY_CONCURRENT_REQUESTS",
                    "retry_after": 1,
                    "timestamp": time.time()
                },
                headers={"Retry-After": "1"}
            )
        
        # Acquire semaphore
        async with self.semaphore:
            self.active_requests += 1
            
            try:
                response = await call_next(request)
                
                # Add concurrency info to headers
                response.headers["X-Active-Requests"] = str(self.active_requests)
                response.headers["X-Max-Concurrent-Requests"] = str(self.settings.max_concurrent_requests)
                
                return response
                
            finally:
                self.active_requests -= 1


# Middleware factory functions

def create_request_timing_middleware() -> RequestTimingMiddleware:
    """Create request timing middleware"""
    return RequestTimingMiddleware


def create_rate_limit_middleware() -> RateLimitMiddleware:
    """Create rate limit middleware"""
    return RateLimitMiddleware


def create_security_headers_middleware() -> SecurityHeadersMiddleware:
    """Create security headers middleware"""
    return SecurityHeadersMiddleware


def create_compression_middleware() -> CompressionMiddleware:
    """Create compression middleware"""
    return CompressionMiddleware


def create_cache_control_middleware() -> CacheControlMiddleware:
    """Create cache control middleware"""
    return CacheControlMiddleware


def create_request_size_middleware() -> RequestSizeMiddleware:
    """Create request size middleware"""
    return RequestSizeMiddleware


def create_concurrency_limit_middleware() -> ConcurrencyLimitMiddleware:
    """Create concurrency limit middleware"""
    return ConcurrencyLimitMiddleware


# Export all middleware classes and factories
__all__ = [
    # Middleware classes
    'RequestTimingMiddleware',
    'RateLimitMiddleware',
    'SecurityHeadersMiddleware',
    'CompressionMiddleware',
    'CacheControlMiddleware',
    'RequestSizeMiddleware',
    'ConcurrencyLimitMiddleware',
    
    # Factory functions
    'create_request_timing_middleware',
    'create_rate_limit_middleware',
    'create_security_headers_middleware',
    'create_compression_middleware',
    'create_cache_control_middleware',
    'create_request_size_middleware',
    'create_concurrency_limit_middleware',
]






























