"""
Advanced Middleware for Content Redundancy Detector
- Enhanced logging with structured data
- CORS configuration for frontend
- Better error handling with frontend-friendly responses
- Performance monitoring with detailed metrics
- Rate limiting with per-endpoint configuration
"""

import time
import logging
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os

try:
    from metrics import record_request_metric
except ImportError:
    # Fallback if metrics module not available
    def record_request_metric(*args, **kwargs):
        pass

try:
    from rate_limiter import check_rate_limit
except ImportError:
    # Fallback if rate_limiter module not available
    def check_rate_limit(*args, **kwargs):
        return True, {"limit": 100, "remaining": 99, "reset_time": time.time() + 60, "retry_after": 0}

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for request/response logging and metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request with structured data
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {client_ip} - {request.headers.get('user-agent', 'unknown')}"
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed: {str(e)} - {process_time:.4f}s",
                exc_info=True
            )
            raise
        
        # Log response and record metrics
        process_time = time.time() - start_time
        logger.info(
            f"Response {request_id}: {response.status_code} - {process_time:.4f}s "
            f"({request.method} {request.url.path})"
        )
        
        # Record metrics
        try:
            record_request_metric(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=process_time
            )
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
        
        # Add performance and tracing headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{process_time:.4f}s"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for global error handling with frontend-friendly responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException as exc:
            # Re-raise HTTP exceptions to be handled by FastAPI
            raise exc
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            
            # Check environment to determine error detail visibility
            environment = os.getenv("ENVIRONMENT", "development").lower()
            is_debug = environment == "development" or os.getenv("DEBUG", "false").lower() == "true"
            
            request_id = getattr(request.state, "request_id", "unknown")
            
            # Frontend-friendly error format
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "message": str(e) if is_debug else "Internal server error",
                        "code": 500,
                        "type": type(e).__name__,
                        "request_id": request_id
                    },
                    "timestamp": time.time()
                },
                headers={
                    "X-Request-ID": request_id,
                    "Content-Type": "application/json"
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for security headers with CORS support"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add comprehensive security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy (can be customized)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response


class CORSMiddlewareWrapper:
    """CORS middleware configuration for frontend integration"""
    
    @staticmethod
    def get_cors_config() -> Dict[str, Any]:
        """Get CORS configuration"""
        # Get allowed origins from environment or use defaults
        cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
        cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]
        
        if not cors_origins:
            # Default development origins
            cors_origins = [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:5173",  # Vite
                "http://localhost:4200",  # Angular
                "http://localhost:8080",  # Vue CLI
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ]
        
        # In development, allow all origins if explicitly set
        environment = os.getenv("ENVIRONMENT", "development").lower()
        if environment == "development" and os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
            cors_origins = ["*"]
        
        return {
            "allow_origins": cors_origins if "*" not in cors_origins else ["*"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
            "allow_headers": [
                "Authorization",
                "Content-Type",
                "Accept",
                "Origin",
                "X-Requested-With",
                "X-User-Id",
                "User-Id",
                "X-API-Key",
                "*"
            ],
            "expose_headers": [
                "Content-Disposition",
                "Content-Type",
                "X-Total-Count",
                "X-Request-ID",
                "X-Response-Time",
                "X-Process-Time"
            ],
            "max_age": 3600,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        is_allowed, rate_info = check_rate_limit(client_ip, request.url.path)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip} on {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Try again in {rate_info['retry_after']} seconds",
                    "retry_after": rate_info['retry_after']
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info['limit']),
                    "X-RateLimit-Remaining": str(rate_info['remaining']),
                    "X-RateLimit-Reset": str(rate_info['reset_time']),
                    "Retry-After": str(rate_info['retry_after'])
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(rate_info['reset_time'])
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{process_time:.4f}s"
        response.headers["X-Request-ID"] = str(int(start_time * 1000))
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(f"Slow request: {request.method} {request.url.path} - {process_time:.4f}s")
        
        return response


# Enhanced request context middleware (optional integration)
try:
    from api.improvements import enhance_request_context
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    def enhance_request_context(request: Request):
        """Fallback if enhancements not available"""
        pass


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to enhance request context with structured logging"""
    
    async def dispatch(self, request: Request, call_next):
        """Enhance request context before processing"""
        if ENHANCEMENTS_AVAILABLE:
            enhance_request_context(request)
        
        return await call_next(request)

