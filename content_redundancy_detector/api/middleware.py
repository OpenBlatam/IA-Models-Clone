"""
Modular Middleware Configuration
Following microservices best practices
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import hashlib

try:
    from ..core.config import get_settings
    from ..core.logging_config import get_logger
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Fallback
    class Settings:
        cors_origins = ["*"]
        log_level = "INFO"
    def get_settings():
        return Settings()
    logger = logging.getLogger(__name__)
    def get_logger(name):
        return logging.getLogger(name)

if CORE_AVAILABLE:
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Import structured logging utilities
try:
    from ..utils.structured_logging import (
        set_request_context, clear_request_context, log_performance
    )
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False
    def set_request_context(*args, **kwargs):
        pass
    def clear_request_context():
        pass
    def log_performance(*args, **kwargs):
        pass


def setup_cors_middleware(app) -> None:
    """Setup CORS middleware"""
    settings = get_settings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging with context tracking"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Set request context for structured logging
        if STRUCTURED_LOGGING_AVAILABLE:
            user_id = request.headers.get("X-User-Id") or request.headers.get("User-Id")
            correlation_id = request.headers.get("X-Correlation-ID")
            set_request_context(
                request_id=request_id,
                user_id=user_id,
                correlation_id=correlation_id
            )
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"Request: {request.method} {request.url.path} from {client_ip}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent"),
                "request_id": request_id
            }
        )
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown"
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.4f}s",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        return response


def setup_logging_middleware(app) -> None:
    """Setup logging middleware"""
    app.add_middleware(LoggingMiddleware)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-IP tracking"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.rate_limits: dict = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Get or create rate limit bucket
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old entries (older than 1 minute)
        window_start = now - 60
        self.rate_limits[client_ip] = [
            t for t in self.rate_limits[client_ip] if t > window_start
        ]
        
        # Check rate limit
        if len(self.rate_limits[client_ip]) >= self.requests_per_minute:
            retry_after = 60 - int(now - self.rate_limits[client_ip][0])
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Try again in {retry_after} seconds",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Record request
        self.rate_limits[client_ip].append(now)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.rate_limits[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))
        
        return response


def setup_rate_limiting_middleware(app) -> None:
    """Setup rate limiting middleware"""
    settings = get_settings()
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=settings.rate_limit_requests
    )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Strong CSP with safe defaults; can be overridden via settings if provided
        try:
            settings = get_settings()
            csp = getattr(settings, "content_security_policy", None)
        except Exception:
            csp = None
        response.headers["Content-Security-Policy"] = csp or (
            "default-src 'self'; frame-ancestors 'none'; object-src 'none'; "
            "img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "base-uri 'self'"
        )
        # HSTS only if behind HTTPS (can be toggled via settings)
        try:
            settings = get_settings()
            enable_hsts = getattr(settings, "enable_hsts", True)
        except Exception:
            enable_hsts = True
        if enable_hsts and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        # Modern isolation/resource policies
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), usb=(), payment=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        
        return response


def setup_security_middleware(app) -> None:
    """Setup security middleware"""
    app.add_middleware(SecurityMiddleware)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Add performance header
        response.headers["X-Response-Time"] = f"{process_time:.4f}s"
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} - {process_time:.4f}s",
                extra={
                    "process_time": process_time,
                    "method": request.method,
                    "path": request.url.path
                }
            )
        
        return response


def setup_performance_middleware(app) -> None:
    """Setup performance middleware"""
    app.add_middleware(PerformanceMiddleware)


class ETagMiddleware(BaseHTTPMiddleware):
    """ETag support for GET/HEAD JSON responses with conditional requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method not in ("GET", "HEAD"):
            return await call_next(request)
        
        response = await call_next(request)
        
        try:
            # Only for JSON responses with a concrete body
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return response
            
            body_bytes: bytes
            if hasattr(response, "body") and response.body is not None:
                body_bytes = response.body  # type: ignore[attr-defined]
            else:
                # Fallback: try to get body from render() if available
                if hasattr(response, "render") and callable(getattr(response, "render")):
                    body_bytes = await response.render(None)  # type: ignore
                else:
                    return response
            
            # Compute strong ETag (SHA256)
            etag = 'W/"' + hashlib.sha256(body_bytes).hexdigest() + '"'
            response.headers["ETag"] = etag
            
            inm = request.headers.get("if-none-match")
            if inm and inm == etag:
                # Not modified
                not_modified = Response(status_code=304)
                not_modified.headers.update({
                    k: v for k, v in response.headers.items()
                    if k.lower() in ("etag", "cache-control", "expires")
                })
                return not_modified
            
            # Default cache headers (override via route if needed)
            response.headers.setdefault("Cache-Control", "public, max-age=30")
        except Exception:
            # Never break the response due to caching
            return response
        
        return response


def setup_http_cache_middleware(app) -> None:
    """Setup HTTP caching (ETag) middleware."""
    app.add_middleware(ETagMiddleware)

