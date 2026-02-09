"""
Middleware Components
====================

This module contains middleware components for the FastAPI application,
including error handling, logging, authentication, and rate limiting.
"""

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from typing import Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        """Log requests and responses"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"📥 {request.method} {request.url.path} - "
            f"Client: {request.client.host} - "
            f"Request ID: {request_id}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"📤 {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s - "
                f"Request ID: {request_id}"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"❌ {request.method} {request.url.path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s - "
                f"Request ID: {request_id}"
            )
            raise


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors globally"""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "HTTP Error",
                    "message": e.detail,
                    "status_code": e.status_code,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except ValueError as e:
            # Handle validation errors
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "status_code": 400,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "status_code": 500,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication (placeholder implementation)"""
    
    def __init__(self, app: ASGIApp, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key or "your-api-key-here"
    
    async def dispatch(self, request: Request, call_next):
        """Handle authentication"""
        # Skip authentication for certain paths
        if request.url.path in ["/", "/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication Required",
                    "message": "API key is required",
                    "status_code": 401,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        if api_key != self.api_key:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Authentication Failed",
                    "message": "Invalid API key",
                    "status_code": 403,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Add user info to request state (placeholder)
        request.state.user_id = "authenticated_user"
        request.state.user_role = "user"
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting (simple implementation)"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        """Handle rate limiting"""
        # Get client IP
        client_ip = request.client.host
        
        # Clean old requests
        current_time = time.time()
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # Keep only last minute
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate Limit Exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute} per minute",
                    "status_code": 429,
                    "timestamp": datetime.utcnow().isoformat(),
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching"""
    
    def __init__(self, app: ASGIApp, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, tuple] = {}  # (response, timestamp)
    
    async def dispatch(self, request: Request, call_next):
        """Handle response caching"""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Create cache key
        cache_key = f"{request.url.path}?{request.url.query}"
        
        # Check cache
        current_time = time.time()
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                # Return cached response
                response.headers["X-Cache"] = "HIT"
                return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = (response, current_time)
            response.headers["X-Cache"] = "MISS"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "endpoints": {}
        }
    
    async def dispatch(self, request: Request, call_next):
        """Collect metrics"""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["response_times"].append(process_time)
        
        if 200 <= response.status_code < 400:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update endpoint metrics
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics["endpoints"]:
            self.metrics["endpoints"][endpoint] = {
                "count": 0,
                "avg_time": 0,
                "errors": 0
            }
        
        endpoint_metrics = self.metrics["endpoints"][endpoint]
        endpoint_metrics["count"] += 1
        endpoint_metrics["avg_time"] = (
            (endpoint_metrics["avg_time"] * (endpoint_metrics["count"] - 1) + process_time) /
            endpoint_metrics["count"]
        )
        
        if response.status_code >= 400:
            endpoint_metrics["errors"] += 1
        
        # Add metrics to response headers
        response.headers["X-Total-Requests"] = str(self.metrics["total_requests"])
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        # Calculate average response time
        avg_response_time = (
            sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            if self.metrics["response_times"] else 0
        )
        
        return {
            **self.metrics,
            "avg_response_time": avg_response_time,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            )
        }




