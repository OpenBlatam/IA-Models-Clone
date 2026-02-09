from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import json
import logging
from typing import Callable, Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp
import uuid
from ..models.fastapi_models import ErrorResponse
from typing import Any, List, Dict, Optional
import asyncio
"""
FastAPI Middleware - Best Practices

This module implements FastAPI middleware following official documentation
best practices for middleware order, error handling, performance monitoring,
and security.
"""


# Import models

# Logger
logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
start_time_var: ContextVar[Optional[float]] = ContextVar("start_time", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# =============================================================================
# CUSTOM MIDDLEWARE CLASSES
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware following FastAPI best practices.
    
    Logs all requests and responses with structured data including:
    - Request ID for tracing
    - Request/response timing
    - User information
    - Request details
    - Response status and size
    """
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
        self.logger = logging.getLogger("request_logger")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Record start time
        start_time = time.time()
        start_time_var.set(start_time)
        
        # Extract user information
        user_id = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)
        user_id_var.set(user_id)
        
        # Log request
        self.log_request(request, request_id, user_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate timing
            process_time = time.time() - start_time
            
            # Log response
            self.log_response(response, request_id, user_id, process_time)
            
            # Add headers for tracing
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            self.log_error(request, request_id, user_id, e, process_time)
            raise
    
    def log_request(self, request: Request, request_id: str, user_id: Optional[str]):
        """Log request details."""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request"
        }
        
        # Remove sensitive headers
        sensitive_headers = {"authorization", "cookie", "x-api-key"}
        log_data["headers"] = {
            k: v for k, v in log_data["headers"].items() 
            if k.lower() not in sensitive_headers
        }
        
        self.logger.info("Request received", extra=log_data)
    
    def log_response(self, response: Response, request_id: str, user_id: Optional[str], process_time: float):
        """Log response details."""
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "user_id": user_id,
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response"
        }
        
        self.logger.info("Response sent", extra=log_data)
    
    def log_error(self, request: Request, request_id: str, user_id: Optional[str], error: Exception, process_time: float):
        """Log error details."""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "user_id": user_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error"
        }
        
        self.logger.error("Request error", extra=log_data, exc_info=True)

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring middleware.
    
    Tracks performance metrics including:
    - Response times
    - Request counts
    - Error rates
    - Resource usage
    """
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
        self.logger = logging.getLogger("performance_logger")
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "error_count": 0,
            "slow_requests": 0
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Update metrics
            self.update_metrics(process_time, response.status_code)
            
            # Log slow requests
            if process_time > 1.0:  # 1 second threshold
                self.log_slow_request(request, process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            self.update_metrics(process_time, 500, is_error=True)
            raise
    
    def update_metrics(self, process_time: float, status_code: int, is_error: bool = False):
        """Update performance metrics."""
        self.metrics["request_count"] += 1
        self.metrics["total_response_time"] += process_time
        
        if is_error or status_code >= 400:
            self.metrics["error_count"] += 1
        
        if process_time > 1.0:
            self.metrics["slow_requests"] += 1
    
    def log_slow_request(self, request: Request, process_time: float):
        """Log slow request details."""
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.warning("Slow request detected", extra=log_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        avg_response_time = (
            self.metrics["total_response_time"] / self.metrics["request_count"]
            if self.metrics["request_count"] > 0 else 0
        )
        
        error_rate = (
            self.metrics["error_count"] / self.metrics["request_count"] * 100
            if self.metrics["request_count"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "average_response_time": avg_response_time,
            "error_rate_percentage": error_rate
        }

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware.
    
    Provides comprehensive error handling including:
    - Exception capture and logging
    - Structured error responses
    - Error categorization
    - Error reporting
    """
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
        self.logger = logging.getLogger("error_logger")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle errors and provide structured responses."""
        try:
            return await call_next(request)
            
        except Exception as e:
            # Log the error
            self.log_error(request, e)
            
            # Create structured error response
            error_response = self.create_error_response(request, e)
            
            return JSONResponse(
                status_code=error_response.error_code,
                content=error_response.model_dump()
            )
    
    def log_error(self, request: Request, error: Exception):
        """Log error with context."""
        request_id = request_id_var.get()
        user_id = user_id_var.get()
        
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "user_id": user_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.error("Unhandled exception", extra=log_data, exc_info=True)
    
    def create_error_response(self, request: Request, error: Exception) -> ErrorResponse:
        """Create structured error response."""
        # Determine error type and status code
        if isinstance(error, ValueError):
            status_code = 400
            message = "Validation error"
        elif isinstance(error, PermissionError):
            status_code = 403
            message = "Permission denied"
        elif isinstance(error, FileNotFoundError):
            status_code = 404
            message = "Resource not found"
        else:
            status_code = 500
            message = "Internal server error"
        
        return ErrorResponse(
            status="error",
            message=message,
            error_code=status_code,
            details=[{
                "field": None,
                "message": str(error),
                "code": type(error).__name__
            }],
            timestamp=datetime.utcnow()
        )

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware.
    
    Adds security headers to all responses including:
    - Content Security Policy
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer Policy
    """
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
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
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self';"
        )
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Implements rate limiting based on:
    - IP address
    - User ID
    - Endpoint
    - Time windows
    """
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        
    """__init__ function."""
super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.logger = logging.getLogger("rate_limit_logger")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting."""
        # Get client identifier
        client_id = self.get_client_id(request)
        
        # Check rate limit
        if self.is_rate_limited(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "error_code": 429,
                    "retry_after": 60
                }
            )
        
        # Update request count
        self.update_request_count(client_id)
        
        return await call_next(request)
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID first
        user_id = user_id_var.get()
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Clean old entries
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if timestamp > window_start
            ]
        
        # Check limit
        request_count = len(self.request_counts.get(client_id, []))
        return request_count >= self.requests_per_minute
    
    def update_request_count(self, client_id: str):
        """Update request count for client."""
        current_time = time.time()
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        self.request_counts[client_id].append(current_time)

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Request context middleware.
    
    Manages request context including:
    - Request state
    - Context variables
    - Request metadata
    """
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Manage request context."""
        # Set request state
        request.state.start_time = time.time()
        request.state.request_id = str(uuid.uuid4())
        
        # Add request metadata
        request.state.metadata = {
            "user_agent": request.headers.get("user-agent"),
            "accept_language": request.headers.get("accept-language"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        return await call_next(request)

# =============================================================================
# MIDDLEWARE FACTORY FUNCTIONS
# =============================================================================

def create_cors_middleware(
    allow_origins: list = None,
    allow_credentials: bool = True,
    allow_methods: list = None,
    allow_headers: list = None
) -> CORSMiddleware:
    """
    Create CORS middleware with best practices.
    
    Args:
        allow_origins: List of allowed origins
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        
    Returns:
        CORSMiddleware: Configured CORS middleware
    """
    if allow_origins is None:
        allow_origins = ["http://localhost:3000", "https://yourdomain.com"]
    
    if allow_methods is None:
        allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    if allow_headers is None:
        allow_headers = ["*"]
    
    return CORSMiddleware(
        app=None,  # Will be set by FastAPI
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers
    )

def create_gzip_middleware(minimum_size: int = 1000) -> GZipMiddleware:
    """
    Create GZip middleware with best practices.
    
    Args:
        minimum_size: Minimum response size for compression
        
    Returns:
        GZipMiddleware: Configured GZip middleware
    """
    return GZipMiddleware(
        minimum_size=minimum_size
    )

def create_trusted_host_middleware(allowed_hosts: list = None) -> TrustedHostMiddleware:
    """
    Create trusted host middleware with best practices.
    
    Args:
        allowed_hosts: List of allowed hosts
        
    Returns:
        TrustedHostMiddleware: Configured trusted host middleware
    """
    if allowed_hosts is None:
        allowed_hosts = ["localhost", "127.0.0.1", "yourdomain.com"]
    
    return TrustedHostMiddleware(
        allowed_hosts=allowed_hosts
    )

# =============================================================================
# MIDDLEWARE STACK CONFIGURATION
# =============================================================================

class MiddlewareStack:
    """
    Middleware stack configuration following FastAPI best practices.
    
    Defines the order and configuration of middleware components.
    """
    
    def __init__(self) -> Any:
        self.middleware_list = []
        self.performance_monitor = PerformanceMonitoringMiddleware(None)
    
    def add_middleware(self, middleware_class: type, **kwargs):
        """Add middleware to the stack."""
        self.middleware_list.append((middleware_class, kwargs))
    
    def configure_default_stack(self) -> Any:
        """Configure default middleware stack with best practices."""
        # 1. Trusted Host (security)
        self.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        
        # 2. CORS (cross-origin)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://yourdomain.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"]
        )
        
        # 3. GZip (compression)
        self.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # 4. Request Context (context management)
        self.add_middleware(RequestContextMiddleware)
        
        # 5. Request Logging (logging)
        self.add_middleware(RequestLoggingMiddleware)
        
        # 6. Performance Monitoring (monitoring)
        self.add_middleware(PerformanceMonitoringMiddleware)
        
        # 7. Rate Limiting (security)
        self.add_middleware(RateLimitingMiddleware, requests_per_minute=100)
        
        # 8. Security Headers (security)
        self.add_middleware(SecurityHeadersMiddleware)
        
        # 9. Error Handling (error management)
        self.add_middleware(ErrorHandlingMiddleware)
    
    def get_middleware_config(self) -> list:
        """Get middleware configuration for FastAPI app."""
        return self.middleware_list
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from monitoring middleware."""
        return self.performance_monitor.get_metrics()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()

async def get_request_start_time() -> Optional[float]:
    """Get request start time from context."""
    return start_time_var.get()

def get_current_user_id() -> Optional[str]:
    """Get current user ID from context."""
    return user_id_var.get()

def log_request_event(event_type: str, **kwargs):
    """Log request event with context."""
    logger = logging.getLogger("request_logger")
    
    log_data = {
        "request_id": get_request_id(),
        "user_id": get_current_user_id(),
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    logger.info(f"Request event: {event_type}", extra=log_data)

# =============================================================================
# MIDDLEWARE REGISTRY
# =============================================================================

# Export middleware classes and utilities
__all__ = [
    # Middleware classes
    "RequestLoggingMiddleware",
    "PerformanceMonitoringMiddleware", 
    "ErrorHandlingMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware",
    "RequestContextMiddleware",
    
    # Factory functions
    "create_cors_middleware",
    "create_gzip_middleware",
    "create_trusted_host_middleware",
    
    # Stack configuration
    "MiddlewareStack",
    
    # Utility functions
    "get_request_id",
    "get_request_start_time", 
    "get_current_user_id",
    "log_request_event"
] 