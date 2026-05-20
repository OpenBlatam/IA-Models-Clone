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

import time
import uuid
import logging
from typing import Callable, Dict, Any
from datetime import datetime, timezone
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from .utils import create_error_response, format_duration_human_readable
from typing import Any, List, Dict, Optional
import asyncio
"""
FastAPI Middleware for Instagram Captions API.

Performance monitoring, logging, security, and error handling middleware.
"""




logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract request info
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "timestamp": timestamp.isoformat()
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Log successful response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "duration_human": format_duration_human_readable(duration)
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            
            # Log error
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2),
                    "duration_human": format_duration_human_readable(duration)
                },
                exc_info=True
            )
            
            # Return error response
            error_response = create_error_response(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
                headers={"X-Request-ID": request_id}
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics collection."""
    
    def __init__(self, app, slow_request_threshold: float = 2.0):
        
    """__init__ function."""
super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "slow_requests": 0,
            "average_response_time": 0.0,
            "endpoint_metrics": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        endpoint = f"{request.method} {request.url.path}"
        
        # Initialize endpoint metrics if not exists
        if endpoint not in self.metrics["endpoint_metrics"]:
            self.metrics["endpoint_metrics"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "error_count": 0,
                "slow_count": 0
            }
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.perf_counter() - start_time
            self._update_metrics(endpoint, duration, False)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Performance-Tier"] = self._get_performance_tier(duration)
            
            # Log slow requests
            if duration > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected: {endpoint} took {format_duration_human_readable(duration)}"
                )
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self._update_metrics(endpoint, duration, True)
            raise
    
    def _update_metrics(self, endpoint: str, duration: float, is_error: bool):
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        
        endpoint_stats = self.metrics["endpoint_metrics"][endpoint]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += duration
        
        if is_error:
            self.metrics["total_errors"] += 1
            endpoint_stats["error_count"] += 1
        
        if duration > self.slow_request_threshold:
            self.metrics["slow_requests"] += 1
            endpoint_stats["slow_count"] += 1
        
        # Update average response time
        total_time = sum(
            stats["total_time"] 
            for stats in self.metrics["endpoint_metrics"].values()
        )
        self.metrics["average_response_time"] = total_time / self.metrics["total_requests"]
    
    def _get_performance_tier(self, duration: float) -> str:
        """Get performance tier based on response time."""
        if duration < 0.1:
            return "excellent"
        elif duration < 0.5:
            return "good"
        elif duration < 1.0:
            return "acceptable"
        elif duration < 2.0:
            return "slow"
        else:
            return "very-slow"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for basic security headers and validation."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return JSONResponse(
                status_code=413,
                content=create_error_response(
                    error_code="REQUEST_TOO_LARGE",
                    message="Request entity too large"
                ).model_dump()
            )
        
        # Check for common attack patterns in URL
        suspicious_patterns = ["../", "..\\", "<script", "javascript:", "vbscript:"]
        url_path = str(request.url)
        
        if any(pattern in url_path.lower() for pattern in suspicious_patterns):
            logger.warning(f"Suspicious request detected: {url_path}")
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    error_code="SUSPICIOUS_REQUEST",
                    message="Request contains suspicious patterns"
                ).model_dump()
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with security considerations."""
    
    def __init__(self, app, allowed_origins: list = None, allow_credentials: bool = False):
        
    """__init__ function."""
super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, request)
            return response
        
        # Process normal request
        response = await call_next(request)
        self._add_cors_headers(response, request)
        
        return response
    
    def _add_cors_headers(self, response: Response, request: Request):
        """Add CORS headers to response."""
        origin = request.headers.get("origin")
        
        if self.allowed_origins == ["*"] or (origin and origin in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Request-ID, X-API-Key"
        )
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException as e:
            # HTTP exceptions are handled by FastAPI
            raise
        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    error_code="VALIDATION_ERROR",
                    message=str(e),
                    request_id=getattr(request.state, "request_id", None)
                ).model_dump()
            )
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    error_code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    request_id=getattr(request.state, "request_id", None)
                ).model_dump()
            )


class CacheHeaderMiddleware(BaseHTTPMiddleware):
    """Middleware to add appropriate cache headers."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.cache_config = {
            "/health": {"max_age": 60, "public": True},
            "/config": {"max_age": 300, "public": True},
            "/quality-guidelines": {"max_age": 3600, "public": True},
            "/timezones": {"max_age": 3600, "public": True}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Skip caching for errors
        if response.status_code >= 400:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response
        
        # Apply cache headers based on endpoint
        path = request.url.path
        cache_settings = self._get_cache_settings(path)
        
        if cache_settings:
            cache_control = f"max-age={cache_settings['max_age']}"
            if cache_settings.get("public"):
                cache_control += ", public"
            else:
                cache_control += ", private"
            
            response.headers["Cache-Control"] = cache_control
        else:
            # Default: no cache for dynamic content
            response.headers["Cache-Control"] = "no-cache, private"
        
        return response
    
    def _get_cache_settings(self, path: str) -> Dict[str, Any]:
        """Get cache settings for a given path."""
        for pattern, settings in self.cache_config.items():
            if path.startswith(pattern):
                return settings
        return None


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware to add compression recommendations."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add compression hint for large responses
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > 1024:
            response.headers["Vary"] = "Accept-Encoding"
        
        return response 