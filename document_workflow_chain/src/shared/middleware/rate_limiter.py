"""
Rate Limiter Middleware
======================

Advanced rate limiting middleware with:
- Token bucket algorithm
- Sliding window
- Per-user and per-IP limiting
- Redis backend support
- Configurable limits
- Burst handling
"""

from __future__ import annotations
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    limit: int = 0
    window: int = 0


class RateLimitStorage(ABC):
    """Abstract rate limit storage"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data"""
        pass
    
    @abstractmethod
    async def set(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """Set rate limit data"""
        pass
    
    @abstractmethod
    async def increment(self, key: str, ttl: int) -> int:
        """Increment counter and return current value"""
        pass


class InMemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage"""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._counters: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data"""
        async with self._lock:
            return self._data.get(key)
    
    async def set(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """Set rate limit data"""
        async with self._lock:
            self._data[key] = data
            # TTL would be handled by cleanup task in production
    
    async def increment(self, key: str, ttl: int) -> int:
        """Increment counter and return current value"""
        async with self._lock:
            current = self._counters.get(key, 0)
            self._counters[key] = current + 1
            return self._counters[key]


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation
    
    Implements the token bucket algorithm for rate limiting with
    burst handling and configurable refill rates.
    """
    
    def __init__(self, storage: RateLimitStorage):
        self._storage = storage
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def check_limit(
        self, 
        key: str, 
        limit: RateLimit,
        identifier: str = "default"
    ) -> RateLimitResult:
        """Check if request is within rate limit"""
        bucket_key = f"{key}:{identifier}"
        
        async with self._lock:
            current_time = time.time()
            
            # Get or create bucket
            bucket = await self._get_or_create_bucket(bucket_key, limit, current_time)
            
            # Refill tokens
            await self._refill_tokens(bucket, limit, current_time)
            
            # Check if request is allowed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                bucket["last_request"] = current_time
                await self._storage.set(bucket_key, bucket, limit.window_size)
                
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_time=datetime.fromtimestamp(current_time + limit.window_size),
                    limit=limit.requests_per_minute,
                    window=limit.window_size
                )
            else:
                # Calculate retry after
                refill_time = bucket["last_refill"] + (1 / limit.requests_per_minute * 60)
                retry_after = max(0, int(refill_time - current_time))
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=datetime.fromtimestamp(refill_time),
                    retry_after=retry_after,
                    limit=limit.requests_per_minute,
                    window=limit.window_size
                )
    
    async def _get_or_create_bucket(
        self, 
        key: str, 
        limit: RateLimit, 
        current_time: float
    ) -> Dict[str, Any]:
        """Get or create token bucket"""
        bucket = await self._storage.get(key)
        
        if bucket is None:
            bucket = {
                "tokens": limit.requests_per_minute,
                "last_refill": current_time,
                "last_request": current_time
            }
        else:
            # Convert timestamps back to float
            bucket["last_refill"] = float(bucket["last_refill"])
            bucket["last_request"] = float(bucket["last_request"])
        
        return bucket
    
    async def _refill_tokens(
        self, 
        bucket: Dict[str, Any], 
        limit: RateLimit, 
        current_time: float
    ) -> None:
        """Refill tokens based on time elapsed"""
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * (limit.requests_per_minute / 60)
        
        bucket["tokens"] = min(
            limit.requests_per_minute,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = current_time


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation
    
    Implements sliding window algorithm for more precise rate limiting.
    """
    
    def __init__(self, storage: RateLimitStorage):
        self._storage = storage
    
    async def check_limit(
        self, 
        key: str, 
        limit: RateLimit,
        identifier: str = "default"
    ) -> RateLimitResult:
        """Check if request is within rate limit using sliding window"""
        window_key = f"{key}:{identifier}:window"
        current_time = time.time()
        window_start = current_time - limit.window_size
        
        # Get current window data
        window_data = await self._storage.get(window_key)
        
        if window_data is None:
            window_data = {
                "requests": [],
                "window_start": window_start
            }
        
        # Clean old requests
        requests = window_data.get("requests", [])
        requests = [req_time for req_time in requests if req_time > window_start]
        
        # Check if under limit
        if len(requests) < limit.requests_per_minute:
            # Add current request
            requests.append(current_time)
            window_data["requests"] = requests
            window_data["window_start"] = window_start
            
            await self._storage.set(window_key, window_data, limit.window_size)
            
            return RateLimitResult(
                allowed=True,
                remaining=limit.requests_per_minute - len(requests),
                reset_time=datetime.fromtimestamp(requests[0] + limit.window_size) if requests else datetime.fromtimestamp(current_time + limit.window_size),
                limit=limit.requests_per_minute,
                window=limit.window_size
            )
        else:
            # Calculate retry after
            oldest_request = min(requests)
            retry_after = int(oldest_request + limit.window_size - current_time)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp(oldest_request + limit.window_size),
                retry_after=max(0, retry_after),
                limit=limit.requests_per_minute,
                window=limit.window_size
            )


class RateLimiter:
    """
    Main rate limiter class
    
    Provides a unified interface for rate limiting with support for
    different algorithms and storage backends.
    """
    
    def __init__(
        self, 
        storage: Optional[RateLimitStorage] = None,
        algorithm: str = "token_bucket"
    ):
        self._storage = storage or InMemoryRateLimitStorage()
        self._algorithm = algorithm
        
        if algorithm == "token_bucket":
            self._limiter = TokenBucketRateLimiter(self._storage)
        elif algorithm == "sliding_window":
            self._limiter = SlidingWindowRateLimiter(self._storage)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Default rate limits
        self._default_limits: Dict[str, RateLimit] = {
            "default": RateLimit(),
            "api": RateLimit(requests_per_minute=100, requests_per_hour=1000),
            "websocket": RateLimit(requests_per_minute=200, requests_per_hour=2000),
            "create_workflow": RateLimit(requests_per_minute=10, requests_per_hour=100),
            "add_node": RateLimit(requests_per_minute=50, requests_per_hour=500),
            "upload": RateLimit(requests_per_minute=20, requests_per_hour=200),
            "download": RateLimit(requests_per_minute=100, requests_per_hour=1000)
        }
    
    async def check_limit(
        self, 
        key: str, 
        limit_name: str = "default",
        identifier: str = "default"
    ) -> RateLimitResult:
        """Check if request is within rate limit"""
        limit = self._default_limits.get(limit_name, self._default_limits["default"])
        return await self._limiter.check_limit(key, limit, identifier)
    
    async def check_user_limit(
        self, 
        user_id: str, 
        limit_name: str = "default"
    ) -> RateLimitResult:
        """Check rate limit for specific user"""
        return await self.check_limit(f"user:{user_id}", limit_name, user_id)
    
    async def check_ip_limit(
        self, 
        ip_address: str, 
        limit_name: str = "default"
    ) -> RateLimitResult:
        """Check rate limit for specific IP address"""
        return await self.check_limit(f"ip:{ip_address}", limit_name, ip_address)
    
    async def check_endpoint_limit(
        self, 
        endpoint: str, 
        identifier: str,
        limit_name: str = "default"
    ) -> RateLimitResult:
        """Check rate limit for specific endpoint"""
        return await self.check_limit(f"endpoint:{endpoint}", limit_name, identifier)
    
    def set_limit(self, limit_name: str, limit: RateLimit) -> None:
        """Set custom rate limit"""
        self._default_limits[limit_name] = limit
    
    def get_limit(self, limit_name: str) -> Optional[RateLimit]:
        """Get rate limit configuration"""
        return self._default_limits.get(limit_name)
    
    def get_all_limits(self) -> Dict[str, RateLimit]:
        """Get all rate limit configurations"""
        return self._default_limits.copy()
    
    async def reset_limit(self, key: str, identifier: str = "default") -> None:
        """Reset rate limit for specific key"""
        bucket_key = f"{key}:{identifier}"
        await self._storage.set(bucket_key, {}, 0)
    
    async def get_usage_stats(self, key: str, identifier: str = "default") -> Dict[str, Any]:
        """Get usage statistics for specific key"""
        bucket_key = f"{key}:{identifier}"
        data = await self._storage.get(bucket_key)
        
        if data is None:
            return {
                "key": key,
                "identifier": identifier,
                "usage": 0,
                "limit": 0,
                "remaining": 0,
                "reset_time": None
            }
        
        # Calculate usage based on algorithm
        if self._algorithm == "token_bucket":
            usage = data.get("tokens", 0)
            limit = self._default_limits["default"].requests_per_minute
            remaining = max(0, limit - usage)
        else:  # sliding_window
            requests = data.get("requests", [])
            current_time = time.time()
            window_start = current_time - self._default_limits["default"].window_size
            usage = len([req for req in requests if req > window_start])
            limit = self._default_limits["default"].requests_per_minute
            remaining = max(0, limit - usage)
        
        return {
            "key": key,
            "identifier": identifier,
            "usage": usage,
            "limit": limit,
            "remaining": remaining,
            "reset_time": data.get("last_refill", current_time) + self._default_limits["default"].window_size
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# FastAPI dependency
async def get_rate_limiter_dependency() -> RateLimiter:
    """FastAPI dependency for rate limiter"""
    return get_rate_limiter()




