"""
Middleware System for Improved Video-OpusClip API

Comprehensive middleware with:
- Request/response logging
- Performance monitoring
- Security headers
- Rate limiting
- Error handling
- CORS management
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import time
import uuid
import structlog
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

from ..config import settings
from ..monitoring import PerformanceMonitor
from ..error_handling import create_error_response

logger = structlog.get_logger("middleware")

# =============================================================================
# REQUEST ID MIDDLEWARE
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add request ID to request and response."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Add to request state
        request.state.request_id = request_id
        
        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app, performance_monitor: Optional[PerformanceMonitor] = None):
        super().__init__(app)
        self.performance_monitor = performance_monitor
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log request and response details."""
        start_time = time.perf_counter()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            request_id=getattr(request.state, "request_id", None)
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                processing_time=processing_time,
                request_id=getattr(request.state, "request_id", None)
            )
            
            # Record performance metrics
            if self.performance_monitor:
                await self.performance_monitor.record_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=response.status_code,
                    response_time=processing_time
                )
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                processing_time=processing_time,
                request_id=getattr(request.state, "request_id", None)
            )
            
            # Record error metrics
            if self.performance_monitor:
                await self.performance_monitor.record_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=500,
                    response_time=processing_time
                )
            
            raise

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and protection."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        if settings.enable_security_headers:
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Content Security Policy
            if settings.content_security_policy:
                response.headers["Content-Security-Policy"] = settings.content_security_policy
            
            # Strict Transport Security (HTTPS only)
            if request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

# =============================================================================
# RATE LIMITING MIDDLEWARE
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting to requests."""
        if not settings.enable_rate_limiting:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old requests
        current_time = time.time()
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < settings.rate_limit_window
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= settings.rate_limit_requests:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                request_count=len(self.request_counts[client_ip]),
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=429,
                content=create_error_response(
                    error_code="RATE_LIMIT_EXCEEDED",
                    message="Rate limit exceeded. Please try again later.",
                    request_id=getattr(request.state, "request_id", None)
                )
            )
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        return await call_next(request)

# =============================================================================
# ERROR HANDLING MIDDLEWARE
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle errors and create standardized error responses."""
        try:
            return await call_next(request)
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            logger.error(
                "HTTP exception",
                status_code=e.status_code,
                detail=e.detail,
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=create_error_response(
                    error_code=f"HTTP_{e.status_code}",
                    message=str(e.detail),
                    request_id=getattr(request.state, "request_id", None)
                )
            )
        except Exception as e:
            # Handle unexpected exceptions
            logger.error(
                "Unexpected error",
                error=str(e),
                error_type=type(e).__name__,
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    error_code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred. Please try again later.",
                    request_id=getattr(request.state, "request_id", None)
                )
            )

# =============================================================================
# CORS MIDDLEWARE
# =============================================================================

def create_cors_middleware(app) -> CORSMiddleware:
    """Create CORS middleware with proper configuration."""
    return CORSMiddleware(
        app=app,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=["X-Request-ID"],
        max_age=3600
    )

# =============================================================================
# TRUSTED HOST MIDDLEWARE
# =============================================================================

def create_trusted_host_middleware(app) -> TrustedHostMiddleware:
    """Create trusted host middleware for security."""
    if settings.is_production:
        allowed_hosts = ["your-domain.com", "*.your-domain.com"]
    else:
        allowed_hosts = ["*"]
    
    return TrustedHostMiddleware(
        app=app,
        allowed_hosts=allowed_hosts
    )

# =============================================================================
# MIDDLEWARE REGISTRY
# =============================================================================

class MiddlewareRegistry:
    """Registry for managing middleware components."""
    
    def __init__(self):
        self.middleware_stack: List[Callable] = []
        self.performance_monitor: Optional[PerformanceMonitor] = None
    
    def set_performance_monitor(self, monitor: PerformanceMonitor):
        """Set performance monitor for middleware."""
        self.performance_monitor = monitor
    
    def add_middleware(self, middleware_class: type, **kwargs):
        """Add middleware to the stack."""
        self.middleware_stack.append((middleware_class, kwargs))
    
    def apply_middleware(self, app):
        """Apply all registered middleware to the app."""
        # Apply middleware in reverse order (last added is first applied)
        for middleware_class, kwargs in reversed(self.middleware_stack):
            if middleware_class == LoggingMiddleware:
                kwargs["performance_monitor"] = self.performance_monitor
            
            app.add_middleware(middleware_class, **kwargs)
        
        # Apply CORS middleware
        app.add_middleware(create_cors_middleware(app))
        
        # Apply trusted host middleware
        app.add_middleware(create_trusted_host_middleware(app))
    
    def get_middleware_info(self) -> Dict[str, Any]:
        """Get information about registered middleware."""
        return {
            "middleware_count": len(self.middleware_stack),
            "middleware_types": [middleware_class.__name__ for middleware_class, _ in self.middleware_stack],
            "performance_monitor_enabled": self.performance_monitor is not None
        }

# =============================================================================
# MIDDLEWARE FACTORY
# =============================================================================

def create_middleware_registry() -> MiddlewareRegistry:
    """Create and configure middleware registry."""
    registry = MiddlewareRegistry()
    
    # Add core middleware
    registry.add_middleware(RequestIDMiddleware)
    registry.add_middleware(LoggingMiddleware)
    registry.add_middleware(SecurityMiddleware)
    registry.add_middleware(RateLimitMiddleware)
    registry.add_middleware(ErrorHandlingMiddleware)
    
    return registry

# =============================================================================
# MIDDLEWARE UTILITIES
# =============================================================================

def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", None)

def log_request_info(request: Request, message: str, **kwargs):
    """Log request information with request ID."""
    logger.info(
        message,
        request_id=get_request_id(request),
        method=request.method,
        url=str(request.url),
        **kwargs
    )

def log_request_error(request: Request, message: str, error: Exception, **kwargs):
    """Log request error with request ID."""
    logger.error(
        message,
        request_id=get_request_id(request),
        method=request.method,
        url=str(request.url),
        error=str(error),
        error_type=type(error).__name__,
        **kwargs
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RequestIDMiddleware',
    'LoggingMiddleware',
    'SecurityMiddleware',
    'RateLimitMiddleware',
    'ErrorHandlingMiddleware',
    'create_cors_middleware',
    'create_trusted_host_middleware',
    'MiddlewareRegistry',
    'create_middleware_registry',
    'get_request_id',
    'log_request_info',
    'log_request_error'
]






























