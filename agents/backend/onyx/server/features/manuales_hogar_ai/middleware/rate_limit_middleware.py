"""
Rate Limiting Middleware
=========================

Implements rate limiting using Redis or in-memory store.
"""

import time
from typing import Callable, Optional
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as aioredis


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(
        self,
        app: ASGIApp,
        redis_url: Optional[str] = None,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.redis_client: Optional[aioredis.Redis] = None
        self.memory_store: dict = defaultdict(list)
        self._redis_initialized = False

    async def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        if request.client:
            return request.client.host
        return "unknown"

    async def _check_rate_limit_redis(self, key: str) -> tuple[bool, int]:
        """Check rate limit using Redis."""
        if not self.redis_client:
            return True, 0

        try:
            now = time.time()
            minute_key = f"ratelimit:minute:{key}"
            hour_key = f"ratelimit:hour:{key}"

            # Check minute limit
            minute_count = await self.redis_client.get(minute_key)
            if minute_count and int(minute_count) >= self.requests_per_minute:
                return False, 429

            # Check hour limit
            hour_count = await self.redis_client.get(hour_key)
            if hour_count and int(hour_count) >= self.requests_per_hour:
                return False, 429

            # Increment counters
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            await pipe.execute()

            return True, 200

        except Exception:
            # Fallback to memory store if Redis fails
            return await self._check_rate_limit_memory(key)

    async def _check_rate_limit_memory(self, key: str) -> tuple[bool, int]:
        """Check rate limit using in-memory store."""
        now = time.time()
        requests = self.memory_store[key]

        # Remove old requests (older than 1 hour)
        requests[:] = [req_time for req_time in requests if now - req_time < 3600]

        # Check limits
        minute_requests = [req_time for req_time in requests if now - req_time < 60]
        if len(minute_requests) >= self.requests_per_minute:
            return False, 429

        if len(requests) >= self.requests_per_hour:
            return False, 429

        # Add current request
        requests.append(now)

        return True, 200

    async def _ensure_redis(self):
        """Ensure Redis client is initialized."""
        if self._redis_initialized or not self.redis_url:
            return
        
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url, decode_responses=True
            )
            await self.redis_client.ping()
            self._redis_initialized = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to connect to Redis: {e}. Using memory store.")
            self.redis_client = None
            self._redis_initialized = True  # Mark as initialized to avoid retries

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/api/v1/health", "/health", "/", "/metrics"]:
            return await call_next(request)

        # Ensure Redis is initialized
        await self._ensure_redis()

        # Get client identifier
        client_ip = await self._get_client_ip(request)
        api_key = request.headers.get("X-API-Key", "")
        key = f"{client_ip}:{api_key}" if api_key else client_ip

        # Check rate limit
        if self.redis_client:
            allowed, status_code = await self._check_rate_limit_redis(key)
        else:
            allowed, status_code = await self._check_rate_limit_memory(key)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)

        return response


