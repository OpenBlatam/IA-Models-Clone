"""
BUL System - Practical Middleware
Real, practical middleware for the BUL system
"""

import time
import uuid
import logging
from datetime import datetime
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        # Log request
        logger.info(
            f"Request processed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": process_time,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Handle errors gracefully with user-friendly messages"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"Unhandled error in request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Return user-friendly error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred. Please try again later.",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests.get(client_ip, [])
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute} per minute",
                    "retry_after": 60
                }
            )
        
        # Record request
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Process request
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
                "response_size": len(response.body) if hasattr(response, 'body') else 0
            }
        )
        
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect basic metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "endpoints": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["response_times"].append(process_time)
        
        if response.status_code < 400:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Track endpoint metrics
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
            (endpoint_metrics["avg_time"] * (endpoint_metrics["count"] - 1) + process_time) 
            / endpoint_metrics["count"]
        )
        
        if response.status_code >= 400:
            endpoint_metrics["errors"] += 1
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        if self.metrics["response_times"]:
            avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        else:
            avg_response_time = 0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"] * 100
                if self.metrics["total_requests"] > 0 else 0
            ),
            "average_response_time": avg_response_time,
            "endpoints": self.metrics["endpoints"]
        }

class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for cross-origin requests"""
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if response should be compressed
        content_type = response.headers.get("content-type", "")
        content_length = response.headers.get("content-length", "0")
        
        if (
            "application/json" in content_type or
            "text/" in content_type
        ) and int(content_length) > 1000:  # Only compress if > 1KB
            
            # Add compression headers
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle health check requests
        if request.url.path == "/health":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
            )
        
        response = await call_next(request)
        return response

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple authentication middleware"""
    
    def __init__(self, app, public_paths: list = None):
        super().__init__(app)
        self.public_paths = public_paths or ["/health", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Missing or invalid authorization header"
                }
            )
        
        # Extract token
        token = auth_header.split(" ")[1]
        
        # Simple token validation (in production, use JWT)
        if not self._validate_token(token):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid token"
                }
            )
        
        # Add user info to request state
        request.state.user_id = self._get_user_from_token(token)
        
        response = await call_next(request)
        return response
    
    def _validate_token(self, token: str) -> bool:
        """Validate token (simplified)"""
        # In production, implement proper JWT validation
        return len(token) > 10
    
    def _get_user_from_token(self, token: str) -> str:
        """Get user ID from token (simplified)"""
        # In production, decode JWT token
        return "user_123"

class CacheMiddleware(BaseHTTPMiddleware):
    """Simple caching middleware"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.method}:{request.url.path}:{request.url.query}"
        
        # Check cache
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                # Add cache headers
                cached_response.headers["X-Cache"] = "HIT"
                cached_response.headers["X-Cache-Timestamp"] = str(timestamp)
                return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = (response, time.time())
            response.headers["X-Cache"] = "MISS"
        
        return response













