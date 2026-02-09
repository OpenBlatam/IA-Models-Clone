"""
Rate limiter middleware for API rate limiting.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import status
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        """
        Initialize rate limiter.
        
        Args:
            app: ASGI application
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = defaultdict(list)
        self.hour_requests = defaultdict(list)
        self._cleanup_interval = 300  # Clean up every 5 minutes
        self._last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Check rate limits and process request."""
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        
        # Clean up old entries periodically
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_entries()
            self._last_cleanup = current_time
        
        # Check minute limit
        now = datetime.now()
        minute_key = now.replace(second=0, microsecond=0)
        
        # Remove requests older than 1 minute
        self.minute_requests[client_id] = [
            req_time for req_time in self.minute_requests[client_id]
            if (now - req_time).total_seconds() < 60
        ]
        
        if len(self.minute_requests[client_id]) >= self.requests_per_minute:
            logger.warning(
                f"Rate limit exceeded (minute) for {client_id}",
                extra={"client": client_id, "limit": self.requests_per_minute}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                    "retry_after": 60
                }
            )
        
        # Check hour limit
        # Remove requests older than 1 hour
        self.hour_requests[client_id] = [
            req_time for req_time in self.hour_requests[client_id]
            if (now - req_time).total_seconds() < 3600
        ]
        
        if len(self.hour_requests[client_id]) >= self.requests_per_hour:
            logger.warning(
                f"Rate limit exceeded (hour) for {client_id}",
                extra={"client": client_id, "limit": self.requests_per_hour}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {self.requests_per_hour} requests per hour",
                    "retry_after": 3600
                }
            )
        
        # Record request
        self.minute_requests[client_id].append(now)
        self.hour_requests[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        return response
    
    def _cleanup_old_entries(self):
        """Clean up old rate limit entries."""
        now = datetime.now()
        
        # Clean minute requests
        for client_id in list(self.minute_requests.keys()):
            self.minute_requests[client_id] = [
                req_time for req_time in self.minute_requests[client_id]
                if (now - req_time).total_seconds() < 60
            ]
            if not self.minute_requests[client_id]:
                del self.minute_requests[client_id]
        
        # Clean hour requests
        for client_id in list(self.hour_requests.keys()):
            self.hour_requests[client_id] = [
                req_time for req_time in self.hour_requests[client_id]
                if (now - req_time).total_seconds() < 3600
            ]
            if not self.hour_requests[client_id]:
                del self.hour_requests[client_id]




