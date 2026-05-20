"""
Rate Limiting Middleware
Applies rate limiting to API requests
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from ...core.infrastructure.rate_limiter import RateLimiter, RateLimitExceeded

logger = logging.getLogger(__name__)

# Global rate limiter instance
_rate_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=20
        )
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    Limits requests per user/IP to prevent abuse
    """
    
    def __init__(self, app, requests_per_second: float = 10.0, burst_size: int = 20):
        super().__init__(app)
        self.rate_limiter = RateLimiter(
            requests_per_second=requests_per_second,
            burst_size=burst_size
        )
        # Paths to exclude from rate limiting
        self.excluded_paths = {
            "/health",
            "/health/",
            "/health/live",
            "/health/ready",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Get rate limit key (user ID or IP address)
        rate_limit_key = self._get_rate_limit_key(request)
        
        # Check rate limit
        if not await self.rate_limiter.is_allowed(rate_limit_key):
            remaining = await self.rate_limiter.get_remaining(rate_limit_key)
            logger.warning(
                f"Rate limit exceeded for {rate_limit_key} on {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 1
                },
                headers={
                    "X-RateLimit-Limit": str(int(self.rate_limiter.requests_per_second)),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": "1"
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        remaining = await self.rate_limiter.get_remaining(rate_limit_key)
        response.headers["X-RateLimit-Limit"] = str(int(self.rate_limiter.requests_per_second))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Get rate limit key from request
        Prefers user ID from auth, falls back to IP address
        """
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"















