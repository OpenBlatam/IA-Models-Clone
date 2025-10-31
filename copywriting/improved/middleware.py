"""
Custom Middleware
================

Custom middleware for logging, monitoring, and request processing.
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging"""
    
    def __init__(self, app: ASGIApp, include_body: bool = False):
        super().__init__(app)
        self.include_body = include_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[{request_id}] from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"[{request_id}] {response.status_code} - {process_time:.3f}s"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"[{request_id}] - {process_time:.3f}s - {str(e)}"
            )
            
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"- {process_time:.3f}s (threshold: {self.slow_request_threshold}s)"
            )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"Unhandled error in middleware: {request.method} {request.url.path} "
                f"[{request_id}] - {type(e).__name__}: {str(e)}"
            )
            
            # Re-raise the exception to be handled by FastAPI's exception handlers
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # In production, use Redis or similar
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: timestamps for ip, timestamps in self.requests.items()
            if any(ts > current_time - 60 for ts in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [ts for ts in self.requests[client_ip] if ts > current_time - 60]
            if len(recent_requests) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                from fastapi import HTTPException
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]
        
        response = await call_next(request)
        return response






























