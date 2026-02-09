"""Rate limiting middleware."""

from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import settings
from core.exceptions import RateLimitExceededError
from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.utcnow()
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.utcnow()
        
        # Cleanup old entries periodically
        if (now - self.last_cleanup) > self.cleanup_interval:
            self._cleanup_old_entries(now)
            self.last_cleanup = now
        
        # Check rate limit
        client_requests = self.requests[client_ip]
        minute_ago = now - timedelta(minutes=1)
        
        # Remove requests older than 1 minute
        client_requests[:] = [req_time for req_time in client_requests if req_time > minute_ago]
        
        if len(client_requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
            )
        
        # Add current request
        client_requests.append(now)
        
        response = await call_next(request)
        return response
    
    def _cleanup_old_entries(self, now: datetime):
        """Remove entries older than 1 minute."""
        minute_ago = now - timedelta(minutes=1)
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > minute_ago
            ]
            if not self.requests[client_ip]:
                del self.requests[client_ip]

