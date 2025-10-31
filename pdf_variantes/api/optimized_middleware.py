"""
PDF Variantes API - Ultra-Optimized Middleware
Maximum performance middleware with minimal overhead
"""

import time
import gzip
from typing import Optional
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message

from ..utils.response_helpers import get_request_id, set_request_id
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class OptimizedRequestIDMiddleware(BaseHTTPMiddleware):
    """Ultra-fast Request ID middleware with minimal overhead"""
    
    async def dispatch(self, request: Request, call_next):
        """Set request ID with minimal overhead"""
        # Fast path: use existing ID if present
        if not hasattr(request.state, "request_id"):
            set_request_id(request)
        
        response = await call_next(request)
        
        # Fast header setting
        if hasattr(request.state, "request_id"):
            response.headers["X-Request-ID"] = request.state.request_id
        
        return response


class OptimizedLoggingMiddleware(BaseHTTPMiddleware):
    """Optimized logging with conditional logging and performance tracking"""
    
    def __init__(self, app: ASGIApp, log_slow_only: bool = False, slow_threshold: float = 1.0):
        super().__init__(app)
        self.log_slow_only = log_slow_only
        self.slow_threshold = slow_threshold
    
    async def dispatch(self, request: Request, call_next):
        """Optimized logging - only log slow requests if configured"""
        start_time = time.perf_counter()
        request_id = get_request_id(request)
        
        response = await call_next(request)
        
        duration = time.perf_counter() - start_time
        
        # Only log if slow or if logging all requests
        if not self.log_slow_only or duration > self.slow_threshold:
            if duration > self.slow_threshold:
                logger.warning(
                    f"[{request_id}] Slow: {duration:.3f}s {request.method} {request.url.path}"
                )
            else:
                logger.debug(
                    f"[{request_id}] {duration:.3f}s {request.method} {request.url.path}"
                )
        
        # Set headers efficiently
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        return response


class OptimizedCompressionMiddleware(BaseHTTPMiddleware):
    """Optimized compression middleware for large responses"""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 500, compress_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compress_level = compress_level
    
    async def dispatch(self, request: Request, call_next):
        """Compress responses larger than minimum_size"""
        # Skip compression for certain paths or content types
        if self._should_skip_compression(request):
            return await call_next(request)
        
        response = await call_next(request)
        
        # Only compress if response is large enough and not already compressed
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Note: Full compression handling requires response body access
        # For now, this middleware sets up compression headers
        # Actual compression is handled by FastAPI's GZipMiddleware if enabled
        response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_skip_compression(self, request: Request) -> bool:
        """Check if compression should be skipped"""
        # Skip compression for images, videos, already compressed files
        path = request.url.path.lower()
        skip_patterns = [
            ".jpg", ".jpeg", ".png", ".gif", ".webp",  # Images
            ".mp4", ".avi", ".webm",  # Videos
            ".pdf", ".zip", ".gz", ".bz2",  # Already compressed
        ]
        return any(path.endswith(pattern) for pattern in skip_patterns)


class OptimizedRateLimitMiddleware(BaseHTTPMiddleware):
    """Optimized rate limiting with efficient data structures"""
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Use dict with tuple values for better memory efficiency
        self._rate_limit_store: dict = {}
        self._cleanup_counter = 0
    
    async def dispatch(self, request: Request, call_next):
        """Optimized rate limiting with periodic cleanup"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health", "/"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Get or create bucket
        if client_ip not in self._rate_limit_store:
            self._rate_limit_store[client_ip] = []
        
        bucket = self._rate_limit_store[client_ip]
        
        # Clean old entries (fast cutoff check)
        cutoff = now - self.window_seconds
        # Use list comprehension for faster filtering
        bucket[:] = [timestamp for timestamp in bucket if timestamp > cutoff]
        
        # Check rate limit
        if len(bucket) >= self.max_requests:
            from ..utils.response_helpers import create_rate_limit_response
            request_id = get_request_id(request)
            error_response = create_rate_limit_response(
                retry_after=self.window_seconds,
                request_id=request_id
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content=error_response,
                headers={"Retry-After": str(self.window_seconds)}
            )
        
        # Add current timestamp
        bucket.append(now)
        
        # Periodic cleanup (every 100 requests)
        self._cleanup_counter += 1
        if self._cleanup_counter % 100 == 0:
            self._cleanup_old_buckets(now)
        
        return await call_next(request)
    
    def _cleanup_old_buckets(self, now: float):
        """Clean up old buckets to prevent memory leaks"""
        cutoff = now - (self.window_seconds * 2)
        keys_to_remove = [
            ip for ip, bucket in self._rate_limit_store.items()
            if not bucket or (bucket and max(bucket) < cutoff)
        ]
        for key in keys_to_remove:
            self._rate_limit_store.pop(key, None)


class OptimizedSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Optimized security headers with pre-computed headers"""
    
    # Pre-compute headers for better performance
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
    
    HTTPS_SECURITY_HEADERS = {
        **SECURITY_HEADERS,
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers efficiently"""
        response = await call_next(request)
        
        # Select headers based on scheme
        headers = (
            self.HTTPS_SECURITY_HEADERS 
            if request.url.scheme == "https" 
            else self.SECURITY_HEADERS
        )
        
        # Set headers efficiently
        for header, value in headers.items():
            response.headers[header] = value
        
        return response


def setup_optimized_middleware(app: ASGIApp, config: Optional[dict] = None):
    """Setup optimized middleware stack with minimal overhead"""
    if config is None:
        config = {}
    
    # Request ID (first, minimal overhead)
    app.add_middleware(OptimizedRequestIDMiddleware)
    
    # Security headers (early, pre-computed)
    app.add_middleware(OptimizedSecurityHeadersMiddleware)
    
    # Rate limiting (before processing, efficient)
    app.add_middleware(
        OptimizedRateLimitMiddleware,
        max_requests=config.get("max_requests", 100),
        window_seconds=config.get("window_seconds", 60)
    )
    
    # Compression (for large responses)
    app.add_middleware(
        OptimizedCompressionMiddleware,
        minimum_size=config.get("compress_min_size", 500),
        compress_level=config.get("compress_level", 6)
    )
    
    # Logging (last, conditional)
    log_slow_only = config.get("log_slow_only", False)
    slow_threshold = config.get("slow_threshold", 1.0)
    app.add_middleware(
        OptimizedLoggingMiddleware,
        log_slow_only=log_slow_only,
        slow_threshold=slow_threshold
    )

