from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import time
import asyncio
from typing import Callable, Optional, Dict, Any
import orjson
import hashlib
from datetime import datetime, timedelta
import uuid
from prometheus_client import Counter, Histogram, Gauge
import logging
from functools import wraps
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from circuitbreaker import circuit
from .logging import get_logger
from .cache import cache_manager
from .config import settings
            import gzip
    import httpx
from typing import Any, List, Dict, Optional
"""
Advanced Middleware for LinkedIn Posts API
==========================================

High-performance middleware with monitoring, security, and optimization features.
"""



logger = get_logger(__name__)

# Metrics
request_counter = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
concurrent_requests = Gauge('api_concurrent_requests', 'Concurrent requests')
error_counter = Counter('api_errors_total', 'Total API errors', ['method', 'endpoint', 'error_type'])
rate_limit_hits = Counter('api_rate_limit_hits_total', 'Rate limit hits', ['endpoint'])


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    High-performance middleware for request tracking and optimization.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Track concurrent requests
        concurrent_requests.inc()
        
        # Start timing
        start_time = time.time()
        request.state.start_time = start_time
        
        # Add performance headers
        response = None
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Track metrics
            request_counter.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Process-Time"] = f"{duration * 1000:.1f}ms"
            
            # Add server timing header for browser dev tools
            response.headers["Server-Timing"] = f"total;dur={duration * 1000:.1f}"
            
            return response
            
        except Exception as e:
            # Track errors
            error_counter.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            ).inc()
            
            logger.error(
                f"Request failed: {e}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": time.time() - start_time
                }
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        finally:
            concurrent_requests.dec()


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Intelligent caching middleware with ETag support.
    """
    
    CACHEABLE_METHODS = ["GET", "HEAD"]
    CACHE_CONTROL_PATHS = {
        "/api/v2/linkedin-posts": "public, max-age=60",
        "/api/v2/linkedin-posts/health": "no-cache",
        "/api/v2/linkedin-posts/performance": "public, max-age=10"
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET/HEAD requests
        if request.method not in self.CACHEABLE_METHODS:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check if-none-match header
        if_none_match = request.headers.get("if-none-match")
        
        # Try to get from cache
        cached_response = await cache_manager.get(f"response:{cache_key}")
        
        if cached_response:
            cached_data = orjson.loads(cached_response)
            etag = cached_data.get("etag")
            
            # Check ETag match
            if if_none_match and if_none_match == etag:
                return Response(status_code=304, headers={"ETag": etag})
            
            # Return cached response
            return Response(
                content=cached_data["content"],
                status_code=cached_data["status_code"],
                headers=cached_data["headers"],
                media_type=cached_data.get("media_type", "application/json")
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200 and request.method == "GET":
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Generate ETag
            etag = f'"{hashlib.md5(body).hexdigest()}"'
            
            # Prepare cache data
            cache_data = {
                "content": body.decode(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type,
                "etag": etag
            }
            
            # Determine cache duration
            cache_duration = self._get_cache_duration(request.url.path)
            
            # Store in cache
            await cache_manager.set(
                f"response:{cache_key}",
                orjson.dumps(cache_data),
                expire=cache_duration
            )
            
            # Add cache headers
            response.headers["ETag"] = etag
            response.headers["Cache-Control"] = self._get_cache_control(request.url.path)
            
            # Return new response with body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        # Include query parameters in cache key
        query_params = str(sorted(request.query_params.items()))
        
        # Include relevant headers
        accept = request.headers.get("accept", "")
        auth = request.headers.get("authorization", "")[:10]  # First 10 chars only
        
        # Generate key
        key_parts = [
            request.method,
            request.url.path,
            query_params,
            accept,
            auth
        ]
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_duration(self, path: str) -> int:
        """Get cache duration for path."""
        if "/health" in path:
            return 5
        elif "/performance" in path:
            return 10
        elif "/analyze" in path:
            return 300
        else:
            return 60
    
    def _get_cache_control(self, path: str) -> str:
        """Get cache control header for path."""
        for pattern, control in self.CACHE_CONTROL_PATHS.items():
            if path.startswith(pattern):
                return control
        return "public, max-age=60"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with sliding window and distributed support.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        
    """__init__ function."""
super().__init__(app)
        self.calls = calls
        self.period = period
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        key = f"rate_limit:{client_id}:{request.url.path}"
        
        # Sliding window rate limiting
        current_time = time.time()
        window_start = current_time - self.period
        
        # Get current window data
        window_key = f"{key}:window"
        window_data = await cache_manager.get(window_key)
        
        if window_data:
            requests = orjson.loads(window_data)
            # Remove old requests
            requests = [ts for ts in requests if ts > window_start]
        else:
            requests = []
        
        # Check if limit exceeded
        if len(requests) >= self.calls:
            rate_limit_hits.labels(endpoint=request.url.path).inc()
            
            # Calculate retry after
            oldest_request = min(requests) if requests else current_time
            retry_after = int(self.period - (current_time - oldest_request))
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest_request + self.period)),
                    "Retry-After": str(retry_after)
                }
            )
        
        # Add current request
        requests.append(current_time)
        
        # Update cache
        await cache_manager.set(
            window_key,
            orjson.dumps(requests),
            expire=self.period
        )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - len(requests))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get from auth token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                return payload.get("sub", "")
            except:
                pass
        
        # Fall back to IP + User-Agent
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        user_agent = request.headers.get("User-Agent", "")
        return f"{client_ip}:{hashlib.md5(user_agent.encode()).hexdigest()[:8]}"


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware with CORS, headers, and protection features.
    """
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add security headers to response
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Add CSP header if not present
        if "Content-Security-Policy" not in response.headers:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https:;"
            )
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Response compression middleware for bandwidth optimization.
    """
    
    MIN_SIZE = 1024  # Minimum size to compress (1KB)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        
        # Process request
        response = await call_next(request)
        
        # Only compress successful responses
        if response.status_code != 200:
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in ["json", "text", "javascript", "css"]):
            return response
        
        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check size threshold
        if len(body) < self.MIN_SIZE:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress based on accept-encoding
        if "gzip" in accept_encoding:
            compressed = gzip.compress(body)
            
            # Only use if smaller
            if len(compressed) < len(body):
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Vary"] = "Accept-Encoding"
                
                return Response(
                    content=compressed,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
        
        # Return original response
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )


# Circuit breaker for external services
@circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
async def call_external_service(url: str, **kwargs):
    """
    Call external service with circuit breaker pattern.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, **kwargs)
        response.raise_for_status()
        return response


# Export middleware
__all__ = [
    "PerformanceMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "CompressionMiddleware",
    "call_external_service"
] 