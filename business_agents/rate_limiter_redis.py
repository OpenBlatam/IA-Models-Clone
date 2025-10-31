"""Distributed rate limiting using Redis."""
import time
import asyncio
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from redis import asyncio as redis  # type: ignore
except ImportError:
    redis = None  # type: ignore


class DistributedRateLimitMiddleware(BaseHTTPMiddleware):
    """Distributed rate limiting using Redis."""
    
    def __init__(
        self,
        app,
        redis_url: str,
        requests_per_minute: int = 100,
        key_prefix: str = "rate_limit"
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.requests_per_minute = requests_per_minute
        self.key_prefix = key_prefix
        self._redis: Optional[redis.Redis] = None
    
    async def init_redis(self):
        """Initialize Redis connection."""
        if redis is None:
            raise ImportError("redis package is required for distributed rate limiting")
        
        try:
            self._redis = await redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            # Test connection
            await self._redis.ping()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to connect to Redis: {e}")
            self._redis = None
    
    def _get_client_key(self, request: Request) -> str:
        """Generate rate limit key for client."""
        client_ip = request.client.host if request.client else "unknown"
        return f"{self.key_prefix}:{client_ip}"
    
    async def dispatch(self, request: Request, call_next):
        """Check rate limit before processing request."""
        # Lazy initialization of Redis
        if self._redis is None:
            await self.init_redis()
        
        if self._redis is None:
            # Fallback: allow request if Redis unavailable
            return await call_next(request)
        
        try:
            key = self._get_client_key(request)
            now = int(time.time())
            window_start = now - 60  # 1 minute window
            
            # Get current count
            count = await self._redis.zcount(key, window_start, now)
            
            if count >= self.requests_per_minute:
                # Rate limit exceeded
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limited",
                        "message": "Too Many Requests",
                        "retry_after": 60
                    }
                )
            
            # Add current request to sorted set
            await self._redis.zadd(key, {str(now): now})
            # Set expiry
            await self._redis.expire(key, 120)  # 2 minutes
            
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            # On error, allow request through (fail open)
            import logging
            logging.getLogger(__name__).warning(f"Rate limit check failed: {e}")
            return await call_next(request)

