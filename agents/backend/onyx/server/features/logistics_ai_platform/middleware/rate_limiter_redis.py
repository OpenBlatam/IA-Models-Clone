"""
Redis-based rate limiting middleware

This module provides scalable rate limiting using Redis for distributed systems.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Tuple
import time

from utils.cache import cache_service
from utils.logger import logger


class RedisRateLimiter:
    """
    Scalable rate limiter using Redis
    
    Provides distributed rate limiting that works across multiple instances.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize Redis rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._fallback_limiter = None
    
    async def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed
        
        Args:
            key: Rate limit key (e.g., client IP)
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        if not hasattr(cache_service, '_redis_connected') or not cache_service._redis_connected:
            return await self._fallback_check(key)
        
        try:
            redis_key = f"ratelimit:{key}"
            now = time.time()
            window_start = now - self.window_seconds
            
            pipe = cache_service.redis_client.pipeline()
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {str(now): now})
            pipe.expire(redis_key, self.window_seconds)
            results = await pipe.execute()
            
            current_count = results[1]
            
            if current_count >= self.max_requests:
                await cache_service.redis_client.zrem(redis_key, str(now))
                return False, 0
            
            remaining = self.max_requests - current_count - 1
            return True, max(0, remaining)
            
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}, using fallback")
            return await self._fallback_check(key)
    
    async def _fallback_check(self, key: str) -> Tuple[bool, int]:
        """Fallback to in-memory rate limiting"""
        if self._fallback_limiter is None:
            from middleware.rate_limiter import RateLimiter
            self._fallback_limiter = RateLimiter(
                self.max_requests,
                self.window_seconds
            )
        
        result = self._fallback_limiter.is_allowed(key)
        return result


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-based rate limiting middleware
    
    Provides scalable rate limiting for distributed deployments.
    """
    
    def __init__(self, app, rate_limiter: RedisRateLimiter):
        """Initialize middleware"""
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        client_ip = request.client.host if request.client else "unknown"
        client_id = request.headers.get("X-Client-ID", client_ip)
        
        allowed, remaining = await self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + self.rate_limiter.window_seconds
        )
        
        return response

