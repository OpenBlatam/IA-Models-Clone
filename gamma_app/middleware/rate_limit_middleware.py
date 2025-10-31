"""
Gamma App - Rate Limiting Middleware
Advanced rate limiting with multiple strategies
"""

import time
import asyncio
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import redis
import json
import hashlib
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    key_prefix: str = "rate_limit"
    skip_successful_requests: bool = False
    skip_failed_requests: bool = False

@dataclass
class RateLimitInfo:
    """Rate limit information"""
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None

class RateLimitMiddleware:
    """Advanced rate limiting middleware"""
    
    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig):
        self.redis = redis_client
        self.config = config
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from request
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit_key(self, identifier: str, window: str) -> str:
        """Get rate limit key"""
        return f"{self.config.key_prefix}:{identifier}:{window}"
    
    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    async def _fixed_window_rate_limit(
        self,
        identifier: str,
        window_size: int,
        limit: int
    ) -> RateLimitInfo:
        """Fixed window rate limiting"""
        window = int(time.time() // window_size)
        key = self._get_rate_limit_key(identifier, f"fixed:{window}")
        
        async with await self._get_lock(key):
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, window_size)
            
            remaining = max(0, limit - current)
            reset_time = (window + 1) * window_size
            
            return RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=window_size if current > limit else None
            )
    
    async def _sliding_window_rate_limit(
        self,
        identifier: str,
        window_size: int,
        limit: int
    ) -> RateLimitInfo:
        """Sliding window rate limiting"""
        now = time.time()
        window_start = now - window_size
        
        key = self._get_rate_limit_key(identifier, "sliding")
        
        async with await self._get_lock(key):
            # Remove old entries
            await self.redis.zremrangebyscore(key, 0, window_start)
            
            # Add current request
            await self.redis.zadd(key, {str(now): now})
            
            # Count requests in window
            current = await self.redis.zcard(key)
            
            # Set expiration
            await self.redis.expire(key, window_size)
            
            remaining = max(0, limit - current)
            reset_time = int(now + window_size)
            
            return RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=window_size if current > limit else None
            )
    
    async def _token_bucket_rate_limit(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float
    ) -> RateLimitInfo:
        """Token bucket rate limiting"""
        key = self._get_rate_limit_key(identifier, "token_bucket")
        
        async with await self._get_lock(key):
            now = time.time()
            
            # Get current bucket state
            bucket_data = await self.redis.hmget(key, "tokens", "last_refill")
            tokens = float(bucket_data[0] or capacity)
            last_refill = float(bucket_data[1] or now)
            
            # Calculate tokens to add
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)
            
            # Check if request can be processed
            if tokens >= 1:
                tokens -= 1
                remaining = int(tokens)
                retry_after = None
            else:
                remaining = 0
                retry_after = int((1 - tokens) / refill_rate)
            
            # Update bucket state
            await self.redis.hmset(key, {
                "tokens": tokens,
                "last_refill": now
            })
            await self.redis.expire(key, 3600)  # 1 hour expiration
            
            return RateLimitInfo(
                limit=capacity,
                remaining=remaining,
                reset_time=int(now + 3600),
                retry_after=retry_after
            )
    
    async def _leaky_bucket_rate_limit(
        self,
        identifier: str,
        capacity: int,
        leak_rate: float
    ) -> RateLimitInfo:
        """Leaky bucket rate limiting"""
        key = self._get_rate_limit_key(identifier, "leaky_bucket")
        
        async with await self._get_lock(key):
            now = time.time()
            
            # Get current bucket state
            bucket_data = await self.redis.hmget(key, "level", "last_leak")
            level = float(bucket_data[0] or 0)
            last_leak = float(bucket_data[1] or now)
            
            # Calculate leaked amount
            time_passed = now - last_leak
            leaked = time_passed * leak_rate
            level = max(0, level - leaked)
            
            # Check if request can be processed
            if level < capacity:
                level += 1
                remaining = int(capacity - level)
                retry_after = None
            else:
                remaining = 0
                retry_after = int((level - capacity + 1) / leak_rate)
            
            # Update bucket state
            await self.redis.hmset(key, {
                "level": level,
                "last_leak": now
            })
            await self.redis.expire(key, 3600)  # 1 hour expiration
            
            return RateLimitInfo(
                limit=capacity,
                remaining=remaining,
                reset_time=int(now + 3600),
                retry_after=retry_after
            )
    
    async def check_rate_limit(
        self,
        request: Request,
        custom_config: Optional[RateLimitConfig] = None
    ) -> RateLimitInfo:
        """Check rate limit for request"""
        config = custom_config or self.config
        identifier = self._get_client_identifier(request)
        
        # Check different time windows
        rate_limits = []
        
        # Per minute limit
        if config.requests_per_minute > 0:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                rate_limit = await self._fixed_window_rate_limit(
                    identifier, 60, config.requests_per_minute
                )
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                rate_limit = await self._sliding_window_rate_limit(
                    identifier, 60, config.requests_per_minute
                )
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                rate_limit = await self._token_bucket_rate_limit(
                    identifier, config.requests_per_minute, config.requests_per_minute / 60.0
                )
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                rate_limit = await self._leaky_bucket_rate_limit(
                    identifier, config.requests_per_minute, config.requests_per_minute / 60.0
                )
            else:
                raise ValueError(f"Unsupported strategy: {config.strategy}")
            
            rate_limits.append(rate_limit)
        
        # Per hour limit
        if config.requests_per_hour > 0:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                rate_limit = await self._fixed_window_rate_limit(
                    identifier, 3600, config.requests_per_hour
                )
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                rate_limit = await self._sliding_window_rate_limit(
                    identifier, 3600, config.requests_per_hour
                )
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                rate_limit = await self._token_bucket_rate_limit(
                    identifier, config.requests_per_hour, config.requests_per_hour / 3600.0
                )
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                rate_limit = await self._leaky_bucket_rate_limit(
                    identifier, config.requests_per_hour, config.requests_per_hour / 3600.0
                )
            
            rate_limits.append(rate_limit)
        
        # Per day limit
        if config.requests_per_day > 0:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                rate_limit = await self._fixed_window_rate_limit(
                    identifier, 86400, config.requests_per_day
                )
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                rate_limit = await self._sliding_window_rate_limit(
                    identifier, 86400, config.requests_per_day
                )
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                rate_limit = await self._token_bucket_rate_limit(
                    identifier, config.requests_per_day, config.requests_per_day / 86400.0
                )
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                rate_limit = await self._leaky_bucket_rate_limit(
                    identifier, config.requests_per_day, config.requests_per_day / 86400.0
                )
            
            rate_limits.append(rate_limit)
        
        # Return the most restrictive rate limit
        if not rate_limits:
            return RateLimitInfo(limit=0, remaining=0, reset_time=int(time.time()))
        
        most_restrictive = min(rate_limits, key=lambda x: x.remaining)
        return most_restrictive
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Middleware entry point"""
        try:
            # Check rate limit
            rate_limit_info = await self.check_rate_limit(request)
            
            # Add rate limit headers
            response = await call_next(request)
            
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_limit_info.reset_time)
            
            if rate_limit_info.retry_after:
                response.headers["Retry-After"] = str(rate_limit_info.retry_after)
            
            # Check if rate limit exceeded
            if rate_limit_info.remaining <= 0:
                error_response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": rate_limit_info.retry_after,
                        "limit": rate_limit_info.limit,
                        "reset_time": rate_limit_info.reset_time
                    }
                )
                
                # Add headers to error response
                error_response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
                error_response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
                error_response.headers["X-RateLimit-Reset"] = str(rate_limit_info.reset_time)
                
                if rate_limit_info.retry_after:
                    error_response.headers["Retry-After"] = str(rate_limit_info.retry_after)
                
                return error_response
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # If rate limiting fails, allow the request to proceed
            return await call_next(request)
    
    async def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        try:
            status = {
                "identifier": identifier,
                "limits": {}
            }
            
            # Check different time windows
            for window, limit in [
                ("minute", self.config.requests_per_minute),
                ("hour", self.config.requests_per_hour),
                ("day", self.config.requests_per_day)
            ]:
                if limit > 0:
                    if window == "minute":
                        window_size = 60
                    elif window == "hour":
                        window_size = 3600
                    else:  # day
                        window_size = 86400
                    
                    if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                        rate_limit = await self._sliding_window_rate_limit(
                            identifier, window_size, limit
                        )
                    else:
                        rate_limit = await self._fixed_window_rate_limit(
                            identifier, window_size, limit
                        )
                    
                    status["limits"][window] = {
                        "limit": rate_limit.limit,
                        "remaining": rate_limit.remaining,
                        "reset_time": rate_limit.reset_time
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            return {"error": str(e)}
    
    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier"""
        try:
            # Get all rate limit keys for identifier
            pattern = f"{self.config.key_prefix}:{identifier}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            return False
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        try:
            stats = {
                "total_identifiers": 0,
                "active_identifiers": 0,
                "rate_limited_identifiers": 0,
                "strategy": self.config.strategy.value,
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day
                }
            }
            
            # Get all rate limit keys
            pattern = f"{self.config.key_prefix}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                # Count unique identifiers
                identifiers = set()
                for key in keys:
                    parts = key.decode().split(":")
                    if len(parts) >= 3:
                        identifiers.add(f"{parts[1]}:{parts[2]}")
                
                stats["total_identifiers"] = len(identifiers)
                stats["active_identifiers"] = len(identifiers)
                
                # Count rate limited identifiers
                rate_limited = 0
                for identifier in identifiers:
                    status = await self.get_rate_limit_status(identifier)
                    if status.get("limits"):
                        for window_data in status["limits"].values():
                            if window_data.get("remaining", 0) <= 0:
                                rate_limited += 1
                                break
                
                stats["rate_limited_identifiers"] = rate_limited
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            return {"error": str(e)}

























