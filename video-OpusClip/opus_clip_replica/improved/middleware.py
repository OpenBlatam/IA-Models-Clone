"""
Custom Middleware for OpusClip Improved
======================================

Advanced middleware stack for security, performance, and monitoring.
"""

import time
import logging
import uuid
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis

from .schemas import ErrorResponse
from .exceptions import RateLimitError, create_rate_limit_error

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Active HTTP connections')
REQUEST_SIZE = Histogram('http_request_size_bytes', 'HTTP request size', ['method', 'endpoint'])
RESPONSE_SIZE = Histogram('http_response_size_bytes', 'HTTP response size', ['method', 'endpoint'])


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging"""
    
    def __init__(self, app: ASGIApp, include_body: bool = False):
        super().__init__(app)
        self.include_body = include_body
        self.logger = structlog.get_logger("request_logger")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            await self._log_response(request, response, request_id, start_time)
            
            return response
            
        except Exception as e:
            # Log error
            await self._log_error(request, e, request_id, start_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        try:
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if self.include_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    log_data["body_size"] = len(body)
                    if len(body) < 10000:  # Only log small bodies
                        log_data["body"] = body.decode("utf-8")
                except Exception:
                    log_data["body_error"] = "Failed to read body"
            
            self.logger.info("Request received", **log_data)
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_response(self, request: Request, response: Response, request_id: str, start_time: float):
        """Log response"""
        try:
            duration = time.time() - start_time
            
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
                "response_size": response.headers.get("content-length", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("Request completed", **log_data)
            
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, request_id: str, start_time: float):
        """Log error"""
        try:
            duration = time.time() - start_time
            
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_ms": duration * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.error("Request failed", **log_data)
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics"""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            await self._record_metrics(request, response, start_time)
            
            return response
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
    
    async def _record_metrics(self, request: Request, response: Response, start_time: float):
        """Record performance metrics"""
        try:
            duration = time.time() - start_time
            method = request.method
            path = request.url.path
            status_code = response.status_code
            
            # Record request count
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            
            # Record request duration
            REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
            
            # Record request size
            content_length = request.headers.get("content-length", 0)
            if content_length:
                REQUEST_SIZE.labels(method=method, endpoint=path).observe(int(content_length))
            
            # Record response size
            response_size = response.headers.get("content-length", 0)
            if response_size:
                RESPONSE_SIZE.labels(method=method, endpoint=path).observe(int(response_size))
            
            # Log slow requests
            if duration > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected: {method} {path} took {duration:.2f}s",
                    extra={
                        "method": method,
                        "path": path,
                        "duration": duration,
                        "status_code": status_code
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = structlog.get_logger("error_handler")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            return await self._handle_http_exception(request, e)
            
        except Exception as e:
            return await self._handle_generic_exception(request, e)
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        try:
            error_response = ErrorResponse(
                error_code=f"HTTP_{exc.status_code}",
                error_message=exc.detail,
                request_id=getattr(request.state, "request_id", None)
            )
            
            self.logger.warning(
                "HTTP exception occurred",
                status_code=exc.status_code,
                detail=exc.detail,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Failed to handle HTTP exception: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    async def _handle_generic_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle generic exceptions"""
        try:
            error_response = ErrorResponse(
                error_code="INTERNAL_ERROR",
                error_message="Internal server error",
                request_id=getattr(request.state, "request_id", None)
            )
            
            self.logger.error(
                "Unhandled exception occurred",
                error_type=type(exc).__name__,
                error_message=str(exc),
                path=request.url.path,
                method=request.method,
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Failed to handle generic exception: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app: ASGIApp, redis_client: redis.Redis, default_limit: int = 100, window: int = 3600):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.window = window
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = await self._get_client_id(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, request):
            return await self._rate_limit_response()
        
        # Process request
        response = await call_next(request)
        
        # Record request
        await self._record_request(client_id, request)
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from request state
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        """Check if client is within rate limit"""
        try:
            # Get endpoint-specific limit
            limit = await self._get_endpoint_limit(request)
            
            # Get current count
            key = f"rate_limit:{client_id}:{request.url.path}"
            current_count = await self.redis_client.get(key)
            
            if current_count is None:
                return True  # First request
            
            return int(current_count) < limit
            
        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return True  # Allow request on error
    
    async def _get_endpoint_limit(self, request: Request) -> int:
        """Get rate limit for specific endpoint"""
        # Define endpoint-specific limits
        endpoint_limits = {
            "/api/v2/opus-clip/analyze": 10,
            "/api/v2/opus-clip/analyze/upload": 5,
            "/api/v2/opus-clip/generate": 20,
            "/api/v2/opus-clip/export": 15,
            "/api/v2/opus-clip/batch/process": 3,
        }
        
        return endpoint_limits.get(request.url.path, self.default_limit)
    
    async def _record_request(self, client_id: str, request: Request):
        """Record request for rate limiting"""
        try:
            key = f"rate_limit:{client_id}:{request.url.path}"
            
            # Increment counter
            await self.redis_client.incr(key)
            
            # Set expiration if this is the first request
            if await self.redis_client.ttl(key) == -1:
                await self.redis_client.expire(key, self.window)
                
        except Exception as e:
            logger.error(f"Failed to record request: {e}")
    
    async def _rate_limit_response(self) -> JSONResponse:
        """Return rate limit exceeded response"""
        error_response = ErrorResponse(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit exceeded. Please try again later.",
            error_details={
                "retry_after": self.window,
                "limit": self.default_limit,
                "window": self.window
            }
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(),
            headers={"Retry-After": str(self.window)}
        )


class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching"""
    
    def __init__(self, app: ASGIApp, redis_client: redis.Redis, default_ttl: int = 300):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.cacheable_methods = {"GET"}
        self.cacheable_paths = {
            "/api/v2/opus-clip/health",
            "/api/v2/opus-clip/stats",
            "/api/v2/opus-clip/analytics"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if request is cacheable
        if not self._is_cacheable(request):
            return await call_next(request)
        
        # Try to get cached response
        cached_response = await self._get_cached_response(request)
        if cached_response:
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache response if successful
        if response.status_code == 200:
            await self._cache_response(request, response)
        
        return response
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable"""
        return (
            request.method in self.cacheable_methods and
            request.url.path in self.cacheable_paths
        )
    
    async def _get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response"""
        try:
            cache_key = f"cache:{request.method}:{request.url.path}:{hash(str(request.query_params))}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                import json
                data = json.loads(cached_data)
                return JSONResponse(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers=data["headers"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached response: {e}")
            return None
    
    async def _cache_response(self, request: Request, response: Response):
        """Cache response"""
        try:
            cache_key = f"cache:{request.method}:{request.url.path}:{hash(str(request.query_params))}"
            
            # Get response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Prepare cache data
            cache_data = {
                "content": body.decode("utf-8"),
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            # Cache response
            import json
            await self.redis_client.setex(
                cache_key,
                self.default_ttl,
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication"""
    
    def __init__(self, app: ASGIApp, excluded_paths: Optional[set] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v2/opus-clip/health",
            "/metrics"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if path is excluded from authentication
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error_code="AUTHENTICATION_REQUIRED",
                    error_message="Authentication required"
                ).model_dump()
            )
        
        token = auth_header.split(" ")[1]
        
        # Validate token (simplified - in production, use proper JWT validation)
        try:
            user_id = await self._validate_token(token)
            if user_id:
                request.state.user_id = user_id
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(
                        error_code="INVALID_TOKEN",
                        error_message="Invalid authentication token"
                    ).model_dump()
                )
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error_code="TOKEN_VALIDATION_ERROR",
                    error_message="Token validation failed"
                ).model_dump()
            )
    
    async def _validate_token(self, token: str) -> Optional[str]:
        """Validate authentication token"""
        # Simplified token validation - in production, use proper JWT validation
        # This is just a placeholder implementation
        if token == "valid_token":
            return "user_123"
        return None


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with enhanced security"""
    
    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = True,
        max_age: int = 86400
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            return await self._handle_preflight(request)
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        await self._add_cors_headers(request, response)
        
        return response
    
    async def _handle_preflight(self, request: Request) -> Response:
        """Handle CORS preflight requests"""
        origin = request.headers.get("Origin")
        
        if self._is_origin_allowed(origin):
            headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
                "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
                "Access-Control-Max-Age": str(self.max_age)
            }
            
            if self.allow_credentials:
                headers["Access-Control-Allow-Credentials"] = "true"
            
            return Response(status_code=200, headers=headers)
        
        return Response(status_code=403)
    
    async def _add_cors_headers(self, request: Request, response: Response):
        """Add CORS headers to response"""
        origin = request.headers.get("Origin")
        
        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins





























