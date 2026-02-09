"""
API Middleware
==============

Custom middleware for the TruthGPT API.
"""

import time
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .metrics import metrics_collector

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: defaultdict = defaultdict(list)
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now()
        
        with self._lock:
            self._requests[client_ip] = [
                req_time for req_time in self._requests[client_ip]
                if now - req_time < timedelta(minutes=1)
            ]
            
            if len(self._requests[client_ip]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RateLimitExceeded",
                        "detail": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            self._requests[client_ip].append(now)
        
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log information."""
        start_time = time.time()
        
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"Response: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            
            metrics_collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                duration=process_time,
                status_code=response.status_code
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            status_code = 500
            if isinstance(e, HTTPException):
                status_code = e.status_code
            
            logger.error(
                f"Error: {request.method} {request.url.path} - "
                f"Exception: {str(e)} - "
                f"Time: {process_time:.3f}s"
            )
            
            metrics_collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                duration=process_time,
                status_code=status_code
            )
            
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response

