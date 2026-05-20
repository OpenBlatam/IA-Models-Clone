"""
Rate limiting middleware
"""

import time
from typing import Callable
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Simple in-memory rate limiter (replace with Redis in production)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    
    Limits requests per IP address or user
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        if not self._check_rate_limit(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from request if authenticated
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        cutoff_time = current_time - 3600
        self._rate_limit_store[client_id] = [
            timestamp for timestamp in self._rate_limit_store[client_id]
            if timestamp > cutoff_time
        ]
        
        # Check minute limit
        minute_cutoff = current_time - 60
        recent_requests = [
            timestamp for timestamp in self._rate_limit_store[client_id]
            if timestamp > minute_cutoff
        ]
        
        if len(recent_requests) >= self.requests_per_minute:
            return False
        
        # Check hour limit
        if len(self._rate_limit_store[client_id]) >= self.requests_per_hour:
            return False
        
        # Record this request
        self._rate_limit_store[client_id].append(current_time)
        
        return True
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        current_time = time.time()
        minute_cutoff = current_time - 60
        
        recent_requests = [
            timestamp for timestamp in self._rate_limit_store[client_id]
            if timestamp > minute_cutoff
        ]
        
        return max(0, self.requests_per_minute - len(recent_requests))

