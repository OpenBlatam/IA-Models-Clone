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
from typing import Dict, Any, Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
    from .config_v5 import config
    from .metrics_v5 import metrics
    from .schemas_v5 import ErrorResponse
    from config_v5 import config
    from metrics_v5 import metrics
    from schemas_v5 import ErrorResponse
from typing import Any, List, Dict, Optional
import asyncio
"""
Instagram Captions API v5.0 - Middleware Module

Ultra-fast middleware stack with security, rate limiting, and performance monitoring.
"""


try:
except ImportError:
    # Handle standalone execution


# Configure structured logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


class UltraFastAuthMiddleware(BaseHTTPMiddleware):
    """Ultra-fast API key authentication middleware."""
    
    def __init__(self, app, excluded_paths: list = None):
        
    """__init__ function."""
super().__init__(app)
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process authentication for protected endpoints."""
        
        # Skip auth for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Extract API key from Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return self._create_auth_error("Missing Authorization header")
        
        # Validate Bearer token format
        try:
            scheme, api_key = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                return self._create_auth_error("Invalid authorization scheme")
        except ValueError:
            return self._create_auth_error("Invalid authorization format")
        
        # Validate API key
        if api_key not in config.VALID_API_KEYS:
            return self._create_auth_error("Invalid API key")
        
        # Add client info to request state
        request.state.api_key = api_key
        request.state.client_authenticated = True
        
        return await call_next(request)
    
    def _create_auth_error(self, message: str) -> JSONResponse:
        """Create standardized authentication error response."""
        error_response = ErrorResponse.create(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error_response.model_dump()
        )


class UltraFastRateLimitMiddleware(BaseHTTPMiddleware):
    """Ultra-fast rate limiting middleware with sliding window."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.client_requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting per client."""
        
        # Get client identifier (API key or IP)
        client_id = getattr(request.state, 'api_key', request.client.host)
        current_time = time.time()
        
        # Clean old requests and check rate limit
        if self._is_rate_limited(client_id, current_time):
            return self._create_rate_limit_error(client_id)
        
        # Record this request
        self._record_request(client_id, current_time)
        
        return await call_next(request)
    
    def _is_rate_limited(self, client_id: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        if client_id not in self.client_requests:
            return False
        
        # Remove expired requests (sliding window)
        window_start = current_time - config.RATE_LIMIT_WINDOW
        self.client_requests[client_id] = [
            req_time for req_time in self.client_requests[client_id]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        return len(self.client_requests[client_id]) >= config.RATE_LIMIT_REQUESTS
    
    async def _record_request(self, client_id: str, current_time: float) -> None:
        """Record request timestamp for client."""
        if client_id not in self.client_requests:
            self.client_requests[client_id] = []
        
        self.client_requests[client_id].append(current_time)
    
    def _create_rate_limit_error(self, client_id: str) -> JSONResponse:
        """Create rate limit exceeded error response."""
        error_response = ErrorResponse.create(
            message=f"Rate limit exceeded: {config.RATE_LIMIT_REQUESTS} requests per {config.RATE_LIMIT_WINDOW} seconds",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            request_id=f"rate-limit-{client_id}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.model_dump(),
            headers={"Retry-After": str(config.RATE_LIMIT_WINDOW)}
        )


class UltraFastLoggingMiddleware(BaseHTTPMiddleware):
    """Ultra-fast structured logging middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests with structured format and performance tracking."""
        start_time = time.time()
        
        # Generate request ID
        request_id = f"req-{int(start_time * 1000000) % 1000000:06d}"
        request.state.request_id = request_id
        
        # Record request start
        metrics.record_request_start()
        
        # Log request start
        logger.info(f"🚀 Request started: {request_id} {request.method} {request.url.path}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_request_end(
                success=200 <= response.status_code < 400,
                response_time=processing_time
            )
            
            # Log successful response
            logger.info(
                f"✅ Request completed: {request_id} "
                f"Status:{response.status_code} "
                f"Time:{processing_time*1000:.1f}ms"
            )
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time*1000:.3f}ms"
            
            return response
            
        except Exception as e:
            # Calculate processing time for error case
            processing_time = time.time() - start_time
            
            # Record error metrics
            metrics.record_request_end(
                success=False,
                response_time=processing_time
            )
            
            # Log error
            logger.error(
                f"❌ Request failed: {request_id} "
                f"Error:{str(e)} "
                f"Time:{processing_time*1000:.1f}ms"
            )
            
            # Create error response
            error_response = ErrorResponse.create(
                message="Internal server error",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(),
                headers={"X-Request-ID": request_id}
            )


class UltraFastSecurityMiddleware(BaseHTTPMiddleware):
    """Ultra-fast security headers middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to all responses."""
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-API-Version": config.API_VERSION
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class UltraFastCORSMiddleware(BaseHTTPMiddleware):
    """Ultra-fast CORS middleware for cross-origin requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS for ultra-fast API responses."""
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            return Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": ", ".join(config.CORS_METHODS),
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Request-ID",
                    "Access-Control-Max-Age": "3600"
                }
            )
        
        # Process normal request
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = ", ".join(config.CORS_METHODS)
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Request-ID"
        
        return response


# Middleware utilities
class MiddlewareUtils:
    """Utility functions for middleware operations."""
    
    @staticmethod
    def get_client_info(request: Request) -> Dict[str, Any]:
        """Extract client information from request."""
        return {
            "client_host": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "api_key": getattr(request.state, 'api_key', None),
            "request_id": getattr(request.state, 'request_id', "unknown"),
            "authenticated": getattr(request.state, 'client_authenticated', False)
        }
    
    @staticmethod
    def create_middleware_stack(app) -> Any:
        """Create and configure the complete middleware stack."""
        
        # Security middleware (first layer)
        app.add_middleware(UltraFastSecurityMiddleware)
        
        # CORS middleware
        app.add_middleware(UltraFastCORSMiddleware)
        
        # Logging middleware (early to catch all requests)
        app.add_middleware(UltraFastLoggingMiddleware)
        
        # Rate limiting middleware
        app.add_middleware(UltraFastRateLimitMiddleware)
        
        # Authentication middleware (last, so other middleware runs first)
        app.add_middleware(UltraFastAuthMiddleware)
        
        return app


# Export public interface
__all__ = [
    'UltraFastAuthMiddleware',
    'UltraFastRateLimitMiddleware',
    'UltraFastLoggingMiddleware',
    'UltraFastSecurityMiddleware',
    'UltraFastCORSMiddleware',
    'MiddlewareUtils'
]