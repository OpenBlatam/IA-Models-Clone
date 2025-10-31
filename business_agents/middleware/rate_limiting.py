"""
Rate Limiting Middleware
========================

Rate limiting and request throttling middleware.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from ..config import config

logger = logging.getLogger(__name__)

class RateLimitInfo:
    """Rate limit information for a client."""
    
    def __init__(self, requests: int = 0, window_start: float = 0):
        self.requests = requests
        self.window_start = window_start

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, requests_per_window: int = None, window_seconds: int = None):
        super().__init__(app)
        self.requests_per_window = requests_per_window or config.rate_limit_requests
        self.window_seconds = window_seconds or config.rate_limit_window
        self.client_requests: Dict[str, RateLimitInfo] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        
        # Skip rate limiting if disabled
        if not config.rate_limit_enabled:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Clean up old entries periodically
        await self._cleanup_old_entries()
        
        # Check rate limit
        if not await self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_window} per {self.window_seconds} seconds",
                    "retry_after": self._get_retry_after(client_id)
                },
                headers={"Retry-After": str(self._get_retry_after(client_id))}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Include user agent for additional uniqueness
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Create a hash of IP + User-Agent for client identification
        import hashlib
        client_string = f"{client_ip}:{user_agent}"
        return hashlib.md5(client_string.encode()).hexdigest()[:16]
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        current_time = time.time()
        
        # Get or create client info
        if client_id not in self.client_requests:
            self.client_requests[client_id] = RateLimitInfo()
        
        client_info = self.client_requests[client_id]
        
        # Check if window has expired
        if current_time - client_info.window_start >= self.window_seconds:
            # Reset window
            client_info.requests = 1
            client_info.window_start = current_time
            return True
        
        # Check if within limit
        if client_info.requests < self.requests_per_window:
            client_info.requests += 1
            return True
        
        return False
    
    def _get_retry_after(self, client_id: str) -> int:
        """Get retry after seconds for rate limited client."""
        if client_id in self.client_requests:
            client_info = self.client_requests[client_id]
            elapsed = time.time() - client_info.window_start
            return max(1, int(self.window_seconds - elapsed))
        return self.window_seconds
    
    def _add_rate_limit_headers(self, response: Response, client_id: str):
        """Add rate limit headers to response."""
        if client_id in self.client_requests:
            client_info = self.client_requests[client_id]
            remaining = max(0, self.requests_per_window - client_info.requests)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(client_info.window_start + self.window_seconds))
    
    async def _cleanup_old_entries(self):
        """Clean up old client entries to prevent memory leaks."""
        current_time = time.time()
        
        # Only cleanup every cleanup_interval seconds
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove entries older than 2 * window_seconds
        cutoff_time = current_time - (2 * self.window_seconds)
        old_clients = [
            client_id for client_id, info in self.client_requests.items()
            if current_time - info.window_start > cutoff_time
        ]
        
        for client_id in old_clients:
            del self.client_requests[client_id]
        
        self.last_cleanup = current_time
        
        if old_clients:
            logger.debug(f"Cleaned up {len(old_clients)} old rate limit entries")

class AdaptiveRateLimitingMiddleware(RateLimitingMiddleware):
    """Adaptive rate limiting that adjusts based on system load."""
    
    def __init__(self, app, base_requests: int = None, base_window: int = None):
        super().__init__(app, base_requests, base_window)
        self.base_requests = self.requests_per_window
        self.base_window = self.window_seconds
        self.load_factor = 1.0
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit with adaptive adjustment."""
        # Adjust rate limit based on system load
        self._adjust_rate_limit()
        
        # Use parent's rate limit check
        return await super()._check_rate_limit(client_id)
    
    def _adjust_rate_limit(self):
        """Adjust rate limit based on system load."""
        # This is a simplified implementation
        # In a real system, you would monitor CPU, memory, response times, etc.
        
        # For now, we'll use a simple time-based adjustment
        current_hour = time.localtime().tm_hour
        
        # Reduce rate limits during peak hours (9-17)
        if 9 <= current_hour <= 17:
            self.load_factor = 0.7
        else:
            self.load_factor = 1.0
        
        # Apply load factor
        self.requests_per_window = int(self.base_requests * self.load_factor)
        self.window_seconds = int(self.base_window / self.load_factor)

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Middleware for IP whitelisting (bypasses rate limiting)."""
    
    def __init__(self, app, whitelist: Optional[list] = None):
        super().__init__(app)
        self.whitelist = whitelist or []
    
    async def dispatch(self, request: Request, call_next):
        """Check if client IP is whitelisted."""
        client_ip = self._get_client_ip(request)
        
        if client_ip in self.whitelist:
            # Add header to indicate whitelisted status
            response = await call_next(request)
            response.headers["X-RateLimit-Whitelisted"] = "true"
            return response
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
