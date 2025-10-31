"""
Enhanced Middleware for BUL API
===============================

Modern middleware components for logging, monitoring, security, and performance.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

from ..utils import get_logger, log_api_call, monitor_performance
from ..config import get_config

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request logging middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = get_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests with enhanced metadata"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, processing_time, request_id)
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._log_error(request, e, processing_time, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "timestamp": datetime.now().isoformat(),
            "headers": dict(request.headers)
        }
        
        logger.info(f"Request started: {request.method} {request.url.path}", extra=log_data)
    
    async def _log_response(self, request: Request, response: Response, processing_time: float, request_id: str):
        """Log response"""
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"{response.status_code} in {processing_time:.3f}s",
            extra=log_data
        )
    
    async def _log_error(self, request: Request, error: Exception, processing_time: float, request_id: str):
        """Log error"""
        log_data = {
            "request_id": request_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"{type(error).__name__}: {error}",
            extra=log_data
        )

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
            "min_processing_time": float('inf'),
            "status_codes": {},
            "endpoints": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor performance metrics"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(request, response, processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_error_metrics(request, e, processing_time)
            raise
    
    def _update_metrics(self, request: Request, response: Response, processing_time: float):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_requests"]
        )
        
        # Update min/max
        if processing_time < self.metrics["min_processing_time"]:
            self.metrics["min_processing_time"] = processing_time
        if processing_time > self.metrics["max_processing_time"]:
            self.metrics["max_processing_time"] = processing_time
        
        # Update status codes
        status_code = response.status_code
        self.metrics["status_codes"][status_code] = self.metrics["status_codes"].get(status_code, 0) + 1
        
        # Update endpoint metrics
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics["endpoints"]:
            self.metrics["endpoints"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0
            }
        
        endpoint_metrics = self.metrics["endpoints"][endpoint]
        endpoint_metrics["count"] += 1
        endpoint_metrics["total_time"] += processing_time
        endpoint_metrics["avg_time"] = endpoint_metrics["total_time"] / endpoint_metrics["count"]
    
    def _update_error_metrics(self, request: Request, error: Exception, processing_time: float):
        """Update error metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        # Update endpoint metrics for errors
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics["endpoints"]:
            self.metrics["endpoints"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "errors": 0
            }
        
        endpoint_metrics = self.metrics["endpoints"][endpoint]
        endpoint_metrics["count"] += 1
        endpoint_metrics["total_time"] += processing_time
        endpoint_metrics["avg_time"] = endpoint_metrics["total_time"] / endpoint_metrics["count"]
        endpoint_metrics["errors"] = endpoint_metrics.get("errors", 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = get_config()
        self.blocked_ips = set()
        self.rate_limits = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply security checks"""
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for blocked IPs
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "detail": "IP address is blocked"}
            )
        
        # Check for suspicious patterns
        if await self._is_suspicious_request(request):
            logger.warning(f"Suspicious request from {client_ip}: {request.url}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "detail": "Request blocked by security policy"}
            )
        
        # Apply rate limiting
        if not await self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "detail": "Too many requests"}
            )
        
        # Add security headers
        response = await call_next(request)
        self._add_security_headers(response)
        
        return response
    
    async def _is_suspicious_request(self, request: Request) -> bool:
        """Check for suspicious request patterns"""
        # Check for SQL injection patterns
        suspicious_patterns = [
            "union select", "drop table", "delete from", "insert into",
            "script>", "javascript:", "onload=", "onerror="
        ]
        
        url_lower = str(request.url).lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return True
        
        # Check for excessive path length
        if len(request.url.path) > 2000:
            return True
        
        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-for", "x-real-ip"]
        for header in suspicious_headers:
            if header in request.headers and len(request.headers[header]) > 100:
                return True
        
        return False
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP"""
        current_time = time.time()
        window_size = 60  # 1 minute window
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old requests
        self.rate_limits[client_ip] = [
            req_time for req_time in self.rate_limits[client_ip]
            if current_time - req_time < window_size
        ]
        
        # Check if under limit
        max_requests = 100  # requests per minute
        if len(self.rate_limits[client_ip]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[client_ip].append(current_time)
        return True
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced error handling middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_counts = {}
        self.error_threshold = 10  # Alert after 10 errors
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors with enhanced logging and monitoring"""
        try:
            return await call_next(request)
        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            await self._handle_unexpected_error(request, e)
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
    
    async def _handle_unexpected_error(self, request: Request, error: Exception):
        """Handle unexpected errors with enhanced logging"""
        error_type = type(error).__name__
        error_key = f"{request.method}:{request.url.path}:{error_type}"
        
        # Count errors
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error details
        error_data = {
            "error_type": error_type,
            "error_message": str(error),
            "request_method": request.method,
            "request_url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "error_count": self.error_counts[error_key]
        }
        
        logger.error(f"Unexpected error: {error_type} - {error}", extra=error_data)
        
        # Alert if error threshold exceeded
        if self.error_counts[error_key] >= self.error_threshold:
            logger.critical(
                f"Error threshold exceeded for {error_key}: "
                f"{self.error_counts[error_key]} errors"
            )

class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware"""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply compression to responses"""
        response = await call_next(request)
        
        # Check if response should be compressed
        if self._should_compress(request, response):
            response.headers["Content-Encoding"] = "gzip"
        
        return response
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """Determine if response should be compressed"""
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "application/xml",
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return False
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False
        
        return True

class CacheMiddleware(BaseHTTPMiddleware):
    """Response caching middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply caching to GET requests"""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Cache hit for {request.url}")
                return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = (response, time.time())
            logger.info(f"Cached response for {request.url}")
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        return f"{request.method}:{request.url.path}:{hash(str(request.query_params))}"
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Cache cleared")

# Middleware factory functions
def create_middleware_stack(app: ASGIApp) -> ASGIApp:
    """Create complete middleware stack"""
    # Add middleware in reverse order (last added is first executed)
    app = ErrorHandlingMiddleware(app)
    app = SecurityMiddleware(app)
    app = PerformanceMiddleware(app)
    app = RequestLoggingMiddleware(app)
    app = CompressionMiddleware(app)
    app = CacheMiddleware(app)
    
    return app

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics from middleware"""
    # This would need to be implemented with a global metrics store
    return {
        "total_requests": 0,
        "avg_processing_time": 0.0,
        "error_rate": 0.0
    }












