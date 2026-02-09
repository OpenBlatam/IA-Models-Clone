"""
Rate Limit Middleware

Middleware for rate limiting with performance optimization.
"""

import asyncio
from typing import Callable, Dict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..logging_config import get_logger
from ..exceptions import RateLimitError

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting con optimización de rendimiento"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_minute * 2))
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()
        self._lock = asyncio.Lock()
    
    def _cleanup_old_requests(self):
        """Clean up old request records (optimized with deque)"""
        now = datetime.now()
        if now - self.last_cleanup > self.cleanup_interval:
            cutoff = now - timedelta(minutes=1)
            for client_id in list(self.requests.keys()):
                # Remove expired timestamps from deque
                client_deque = self.requests[client_id]
                while client_deque and client_deque[0] < cutoff:
                    client_deque.popleft()
                if not client_deque:
                    del self.requests[client_id]
            self.last_cleanup = now
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get from forwarded header first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting (thread-safe)"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)
        
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        # Get client ID
        client_id = self._get_client_id(request)
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        async with self._lock:
            # Get client deque and filter recent requests
            client_deque = self.requests[client_id]
            # Remove expired entries
            while client_deque and client_deque[0] < cutoff:
                client_deque.popleft()
            
            # Check rate limit
            recent_count = len(client_deque)
            
            if recent_count >= self.requests_per_minute:
                retry_after = 60 - (now - client_deque[0]).seconds if client_deque else 60
                logger.warning(
                    f"Rate limit exceeded for {client_id}",
                    extra={
                        "client_id": client_id,
                        "requests": recent_count,
                        "limit": self.requests_per_minute,
                        "path": request.url.path
                    }
                )
                raise RateLimitError(retry_after=retry_after)
            
            # Record request
            client_deque.append(now)
            remaining = self.requests_per_minute - recent_count - 1
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response

