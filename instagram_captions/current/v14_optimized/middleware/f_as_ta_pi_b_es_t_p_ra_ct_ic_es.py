from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import uuid
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
import asyncio
"""
FastAPI Best Practices - Middleware

This module implements middleware following FastAPI best practices:
- Request/response processing
- Error handling and logging
- Performance monitoring
- Security headers
- CORS handling
- Rate limiting
- Request ID tracking
"""


logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST ID MIDDLEWARE
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request ID to all requests.
    
    Best practices:
    - Unique identifier for request tracking
    - Consistent across all services
    - Easy debugging and monitoring
    """
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        
    """__init__ function."""
super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response


# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware.
    
    Best practices:
    - Structured logging
    - Performance metrics
    - Error tracking
    - Request/response details
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        
    """__init__ function."""
super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract request details
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, request_id, processing_time)
            
            return response
            
        except Exception as e:
            # Log error
            processing_time = time.time() - start_time
            await self._log_error(request, e, request_id, processing_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log request details"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_length": request.headers.get("content-length"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    log_data["request_body"] = body.decode()[:1000]  # Limit body size
            except Exception:
                log_data["request_body"] = "[Unable to read body]"
        
        logger.info(f"Request: {json.dumps(log_data)}")
    
    async def _log_response(self, request: Request, response: Response, request_id: str, processing_time: float):
        """Log response details"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "processing_time": round(processing_time, 3),
            "content_length": response.headers.get("content-length"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.log_response_body and response.status_code < 400:
            try:
                # Note: This is simplified - in practice, you'd need to handle different response types
                log_data["response_body"] = "[Response body logged]"
            except Exception:
                log_data["response_body"] = "[Unable to read response]"
        
        logger.info(f"Response: {json.dumps(log_data)}")
    
    async def _log_error(self, request: Request, error: Exception, request_id: str, processing_time: float):
        """Log error details"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.error(f"Error: {json.dumps(log_data)}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


# =============================================================================
# PERFORMANCE MONITORING MIDDLEWARE
# =============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring middleware.
    
    Best practices:
    - Response time tracking
    - Resource usage monitoring
    - Performance alerts
    - Metrics collection
    """
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        
    """__init__ function."""
super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Add performance tracking to request state
        request.state.performance_start = start_time
        
        try:
            response = await call_next(request)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            
            # Log slow requests
            if processing_time > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected - "
                    f"Method: {request.method}, "
                    f"URL: {request.url}, "
                    f"Time: {processing_time:.3f}s, "
                    f"Threshold: {self.slow_request_threshold}s"
                )
            
            # Add performance headers
            response.headers["X-Processing-Time"] = str(round(processing_time, 3))
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log performance for failed requests
            logger.error(
                f"Request failed - "
                f"Method: {request.method}, "
                f"URL: {request.url}, "
                f"Time: {processing_time:.3f}s, "
                f"Error: {str(e)}"
            )
            raise


# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware.
    
    Best practices:
    - Security headers for protection
    - Content Security Policy
    - XSS protection
    - Clickjacking protection
    """
    
    def __init__(self, app) -> Any:
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response


# =============================================================================
# RATE LIMITING MIDDLEWARE
# =============================================================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Best practices:
    - Per-user rate limiting
    - Sliding window algorithm
    - Configurable limits
    - Proper error responses
    """
    
    def __init__(self, app, requests_per_minute: int = 100, requests_per_hour: int = 1000):
        
    """__init__ function."""
super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_counts: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get user identifier
        user_id = self._get_user_id(request)
        
        # Check rate limits
        if not self._check_rate_limit(user_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Update rate limit counters
        self._update_rate_limit(user_id)
        
        return response
    
    def _get_user_id(self, request: Request) -> str:
        """Get user identifier for rate limiting"""
        # Try to get from authentication
        if hasattr(request.state, 'user') and request.state.user:
            return request.state.user.get('id', 'anonymous')
        
        # Fallback to IP address
        return self._get_client_ip(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        now = time.time()
        
        if user_id not in self.request_counts:
            return True
        
        user_data = self.request_counts[user_id]
        
        # Check minute limit
        minute_requests = [t for t in user_data.get('minute_requests', []) if now - t < 60]
        if len(minute_requests) >= self.requests_per_minute:
            return False
        
        # Check hour limit
        hour_requests = [t for t in user_data.get('hour_requests', []) if now - t < 3600]
        if len(hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    def _update_rate_limit(self, user_id: str):
        """Update rate limit counters"""
        now = time.time()
        
        if user_id not in self.request_counts:
            self.request_counts[user_id] = {
                'minute_requests': [],
                'hour_requests': []
            }
        
        user_data = self.request_counts[user_id]
        
        # Add current request
        user_data['minute_requests'].append(now)
        user_data['hour_requests'].append(now)
        
        # Clean up old entries
        user_data['minute_requests'] = [t for t in user_data['minute_requests'] if now - t < 60]
        user_data['hour_requests'] = [t for t in user_data['hour_requests'] if now - t < 3600]


# =============================================================================
# ERROR HANDLING MIDDLEWARE
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Best practices:
    - Consistent error responses
    - Error logging and monitoring
    - Security considerations
    - User-friendly messages
    """
    
    def __init__(self, app) -> Any:
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
            
        except Exception as e:
            # Get request ID for tracking
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # Log error
            logger.error(
                f"Unhandled error - "
                f"Request ID: {request_id}, "
                f"Method: {request.method}, "
                f"URL: {request.url}, "
                f"Error: {str(e)}"
            )
            
            # Return consistent error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


# =============================================================================
# CACHE CONTROL MIDDLEWARE
# =============================================================================

class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Cache control middleware.
    
    Best practices:
    - Appropriate cache headers
    - ETag support
    - Cache validation
    - Performance optimization
    """
    
    def __init__(self, app) -> Any:
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add cache control headers based on response type
        if request.method == "GET":
            # Cache GET requests for 5 minutes
            response.headers["Cache-Control"] = "public, max-age=300"
            response.headers["ETag"] = self._generate_etag(response)
        elif request.method in ["POST", "PUT", "DELETE"]:
            # No cache for state-changing operations
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response
    
    def _generate_etag(self, response: Response) -> str:
        """Generate ETag for response"""
        # Simplified ETag generation
        content = str(response.body) if hasattr(response, 'body') else str(response)
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# MIDDLEWARE FACTORY
# =============================================================================

def create_middleware_stack() -> list:
    """
    Create middleware stack following FastAPI best practices.
    
    Order is important:
    1. TrustedHostMiddleware (security)
    2. RequestIDMiddleware (tracking)
    3. LoggingMiddleware (monitoring)
    4. PerformanceMonitoringMiddleware (metrics)
    5. RateLimitingMiddleware (protection)
    6. SecurityHeadersMiddleware (security)
    7. CacheControlMiddleware (performance)
    8. ErrorHandlingMiddleware (error handling)
    """
    
    return [
        # Security first
        TrustedHostMiddleware(allowed_hosts=["*"]),  # Configure appropriately for production
        
        # Request tracking
        RequestIDMiddleware,
        
        # Monitoring and logging
        LoggingMiddleware,
        PerformanceMonitoringMiddleware,
        
        # Protection
        RateLimitingMiddleware,
        
        # Security headers
        SecurityHeadersMiddleware,
        
        # Performance
        CacheControlMiddleware,
        
        # Error handling last
        ErrorHandlingMiddleware,
    ]


def create_cors_middleware() -> CORSMiddleware:
    """
    Create CORS middleware with best practices.
    """
    
    return CORSMiddleware(
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Processing-Time"]
    )


def create_gzip_middleware() -> GZipMiddleware:
    """
    Create GZip middleware for compression.
    """
    
    return GZipMiddleware(minimum_size=1000)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')


def get_processing_time(request: Request) -> float:
    """Get processing time from request state"""
    start_time = getattr(request.state, 'start_time', time.time())
    return time.time() - start_time


def log_request_metrics(request: Request, response: Response, processing_time: float):
    """Log request metrics for monitoring"""
    metrics = {
        "request_id": get_request_id(request),
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "processing_time": processing_time,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(f"Metrics: {json.dumps(metrics)}")


# Export middleware components
__all__ = [
    "RequestIDMiddleware",
    "LoggingMiddleware",
    "PerformanceMonitoringMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware",
    "ErrorHandlingMiddleware",
    "CacheControlMiddleware",
    "create_middleware_stack",
    "create_cors_middleware",
    "create_gzip_middleware",
    "get_request_id",
    "get_processing_time",
    "log_request_metrics"
] 