"""
Middleware for Robot Maintenance Teaching AI API.
"""

import time
from typing import Callable, Dict
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        enabled: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.enabled = enabled
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        self.excluded_paths = {
            "/",
            "/health",
            "/api/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP address)."""
        if request.client:
            return request.client.host
        return "unknown"
    
    def _cleanup_old_entries(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_minute = current_time - 60
        cutoff_hour = current_time - 3600
        
        for client_id in list(self.minute_requests.keys()):
            self.minute_requests[client_id] = [
                ts for ts in self.minute_requests[client_id] if ts > cutoff_minute
            ]
            if not self.minute_requests[client_id]:
                del self.minute_requests[client_id]
        
        for client_id in list(self.hour_requests.keys()):
            self.hour_requests[client_id] = [
                ts for ts in self.hour_requests[client_id] if ts > cutoff_hour
            ]
            if not self.hour_requests[client_id]:
                del self.hour_requests[client_id]
        
        self.last_cleanup = current_time
    
    def _check_rate_limit(self, client_id: str) -> tuple[bool, int, int]:
        """
        Check if client has exceeded rate limits.
        Returns: (allowed, remaining_minute, remaining_hour)
        """
        current_time = time.time()
        
        # Clean up old entries
        self._cleanup_old_entries()
        
        # Check minute limit
        minute_window_start = current_time - 60
        self.minute_requests[client_id] = [
            ts for ts in self.minute_requests[client_id] if ts > minute_window_start
        ]
        
        minute_count = len(self.minute_requests[client_id])
        remaining_minute = max(0, self.requests_per_minute - minute_count)
        
        # Check hour limit
        hour_window_start = current_time - 3600
        self.hour_requests[client_id] = [
            ts for ts in self.hour_requests[client_id] if ts > hour_window_start
        ]
        
        hour_count = len(self.hour_requests[client_id])
        remaining_hour = max(0, self.requests_per_hour - hour_count)
        
        # Check if limits exceeded
        allowed = minute_count < self.requests_per_minute and hour_count < self.requests_per_hour
        
        if allowed:
            # Record request
            self.minute_requests[client_id].append(current_time)
            self.hour_requests[client_id].append(current_time)
        
        return allowed, remaining_minute, remaining_hour
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        
        # Skip rate limiting if disabled or for excluded paths
        if not self.enabled or request.url.path in self.excluded_paths:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        allowed, remaining_minute, remaining_hour = self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "limits": {
                        "per_minute": self.requests_per_minute,
                        "per_hour": self.requests_per_hour
                    },
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.requests_per_hour),
                    "X-RateLimit-Remaining-Minute": str(remaining_minute),
                    "X-RateLimit-Remaining-Hour": str(remaining_hour)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining_minute)
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining_hour)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting API metrics.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "requests_by_endpoint": defaultdict(int),
            "requests_by_status": defaultdict(int),
            "total_response_time": 0.0,
            "errors": 0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for requests."""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        self.metrics["requests_by_endpoint"][request.url.path] += 1
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            self.metrics["total_response_time"] += process_time
            self.metrics["requests_by_status"][response.status_code] += 1
            
            if response.status_code >= 400:
                self.metrics["errors"] += 1
            
            # Add metrics header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
        except Exception as e:
            self.metrics["errors"] += 1
            self.metrics["requests_by_status"][500] += 1
            raise
    
    def get_metrics(self) -> Dict:
        """Get collected metrics."""
        avg_response_time = (
            self.metrics["total_response_time"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0.0
        )
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["errors"],
            "average_response_time": avg_response_time,
            "requests_by_endpoint": dict(self.metrics["requests_by_endpoint"]),
            "requests_by_status": dict(self.metrics["requests_by_status"]),
            "timestamp": datetime.now().isoformat()
        }








