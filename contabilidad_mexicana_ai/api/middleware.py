"""
Middleware for Contabilidad Mexicana AI API
===========================================

Rate limiting and request processing middleware.
"""

import time
import logging
from typing import Callable, Dict
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API endpoints."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        
        # Track requests per IP
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        
        # Excluded paths
        self.excluded_paths = {
            "/api/contador/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP address)."""
        if request.client:
            return request.client.host
        return "unknown"
    
    def _clean_old_requests(self, client_id: str) -> None:
        """Remove old requests outside the time windows."""
        now = time.time()
        
        # Clean minute window (last 60 seconds)
        self.minute_requests[client_id] = [
            ts for ts in self.minute_requests[client_id]
            if now - ts < 60
        ]
        
        # Clean hour window (last 3600 seconds)
        self.hour_requests[client_id] = [
            ts for ts in self.hour_requests[client_id]
            if now - ts < 3600
        ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip rate limiting for non-API paths
        if not request.url.path.startswith("/api/contador"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(client_id)
        
        # Check minute limit
        minute_count = len(self.minute_requests[client_id])
        if minute_count >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute) for {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + 60))
                }
            )
        
        # Check hour limit
        hour_count = len(self.hour_requests[client_id])
        if hour_count >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded (hour) for {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_hour} requests per hour",
                    "retry_after": 3600
                },
                headers={
                    "Retry-After": "3600",
                    "X-RateLimit-Limit": str(self.requests_per_hour),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Record request
        self.minute_requests[client_id].append(now)
        self.hour_requests[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining_minute = self.requests_per_minute - minute_count - 1
        remaining_hour = self.requests_per_hour - hour_count - 1
        
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, remaining_minute))
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, remaining_hour))
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Log request details."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} in {duration:.3f}s",
            extra={
                "status_code": response.status_code,
                "duration": duration,
                "path": request.url.path
            }
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        return response
