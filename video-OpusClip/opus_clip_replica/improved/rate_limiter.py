"""
Rate Limiting for OpusClip Improved
==================================

Advanced rate limiting with Redis backend and multiple strategies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

from .schemas import get_settings
from .exceptions import RateLimitError, create_rate_limit_error

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    strategy: RateLimitStrategy
    limit: int
    window: int  # seconds
    burst_limit: Optional[int] = None
    refill_rate: Optional[float] = None  # tokens per second


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    limit: int = 0
    window: int = 0


class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self._initialize_redis()
        
        # Default rate limit configurations
        self.default_configs = {
            "global": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=1000,
                window=3600  # 1 hour
            ),
            "user": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=100,
                window=3600  # 1 hour
            ),
            "ip": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=50,
                window=3600  # 1 hour
            ),
            "endpoint": {
                "/api/v2/opus-clip/analyze": RateLimitConfig(
                    strategy=RateLimitStrategy.TOKEN_BUCKET,
                    limit=10,
                    window=3600,
                    burst_limit=5,
                    refill_rate=0.1
                ),
                "/api/v2/opus-clip/analyze/upload": RateLimitConfig(
                    strategy=RateLimitStrategy.FIXED_WINDOW,
                    limit=5,
                    window=3600
                ),
                "/api/v2/opus-clip/generate": RateLimitConfig(
                    strategy=RateLimitStrategy.SLIDING_WINDOW,
                    limit=20,
                    window=3600
                ),
                "/api/v2/opus-clip/export": RateLimitConfig(
                    strategy=RateLimitStrategy.SLIDING_WINDOW,
                    limit=15,
                    window=3600
                ),
                "/api/v2/opus-clip/batch/process": RateLimitConfig(
                    strategy=RateLimitStrategy.FIXED_WINDOW,
                    limit=3,
                    window=3600
                )
            }
        }
    
    def _initialize_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis for rate limiting: {e}")
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig,
        increment: bool = True
    ) -> RateLimitResult:
        """Check rate limit for identifier"""
        try:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(identifier, config, increment)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(identifier, config, increment)
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(identifier, config, increment)
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._check_leaky_bucket(identifier, config, increment)
            else:
                raise ValueError(f"Unsupported rate limit strategy: {config.strategy}")
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow request on error
            return RateLimitResult(
                allowed=True,
                remaining=config.limit,
                reset_time=datetime.utcnow() + timedelta(seconds=config.window),
                limit=config.limit,
                window=config.window
            )
    
    async def _check_fixed_window(self, identifier: str, config: RateLimitConfig, increment: bool) -> RateLimitResult:
        """Fixed window rate limiting"""
        try:
            # Calculate window start
            now = int(time.time())
            window_start = (now // config.window) * config.window
            
            key = f"rate_limit:fixed:{identifier}:{window_start}"
            
            if increment:
                # Increment counter
                current_count = await self.redis_client.incr(key)
                await self.redis_client.expire(key, config.window)
            else:
                # Just check current count
                current_count = await self.redis_client.get(key) or 0
                current_count = int(current_count)
            
            allowed = current_count <= config.limit
            remaining = max(0, config.limit - current_count)
            reset_time = datetime.fromtimestamp(window_start + config.window)
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                limit=config.limit,
                window=config.window
            )
            
        except Exception as e:
            logger.error(f"Fixed window rate limit check failed: {e}")
            raise
    
    async def _check_sliding_window(self, identifier: str, config: RateLimitConfig, increment: bool) -> RateLimitResult:
        """Sliding window rate limiting"""
        try:
            now = int(time.time())
            window_start = now - config.window
            
            key = f"rate_limit:sliding:{identifier}"
            
            if increment:
                # Add current timestamp
                await self.redis_client.zadd(key, {str(now): now})
                # Remove old entries
                await self.redis_client.zremrangebyscore(key, 0, window_start)
                # Set expiration
                await self.redis_client.expire(key, config.window)
            else:
                # Remove old entries first
                await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count entries in window
            current_count = await self.redis_client.zcard(key)
            
            allowed = current_count <= config.limit
            remaining = max(0, config.limit - current_count)
            reset_time = datetime.fromtimestamp(now + config.window)
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                limit=config.limit,
                window=config.window
            )
            
        except Exception as e:
            logger.error(f"Sliding window rate limit check failed: {e}")
            raise
    
    async def _check_token_bucket(self, identifier: str, config: RateLimitConfig, increment: bool) -> RateLimitResult:
        """Token bucket rate limiting"""
        try:
            now = time.time()
            key = f"rate_limit:bucket:{identifier}"
            
            # Get current bucket state
            bucket_data = await self.redis_client.hgetall(key)
            
            if not bucket_data:
                # Initialize bucket
                tokens = config.burst_limit or config.limit
                last_refill = now
            else:
                tokens = float(bucket_data.get("tokens", config.burst_limit or config.limit))
                last_refill = float(bucket_data.get("last_refill", now))
            
            # Refill tokens
            time_passed = now - last_refill
            refill_rate = config.refill_rate or (config.limit / config.window)
            tokens_to_add = time_passed * refill_rate
            tokens = min(config.burst_limit or config.limit, tokens + tokens_to_add)
            
            if increment:
                if tokens >= 1:
                    tokens -= 1
                    allowed = True
                else:
                    allowed = False
                
                # Update bucket state
                await self.redis_client.hset(key, mapping={
                    "tokens": str(tokens),
                    "last_refill": str(now)
                })
                await self.redis_client.expire(key, config.window)
            else:
                allowed = tokens >= 1
            
            remaining = int(tokens)
            reset_time = datetime.fromtimestamp(now + config.window)
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                limit=config.limit,
                window=config.window
            )
            
        except Exception as e:
            logger.error(f"Token bucket rate limit check failed: {e}")
            raise
    
    async def _check_leaky_bucket(self, identifier: str, config: RateLimitConfig, increment: bool) -> RateLimitResult:
        """Leaky bucket rate limiting"""
        try:
            now = time.time()
            key = f"rate_limit:leaky:{identifier}"
            
            # Get current bucket state
            bucket_data = await self.redis_client.hgetall(key)
            
            if not bucket_data:
                # Initialize bucket
                level = 0
                last_leak = now
            else:
                level = float(bucket_data.get("level", 0))
                last_leak = float(bucket_data.get("last_leak", now))
            
            # Leak water
            time_passed = now - last_leak
            leak_rate = config.limit / config.window
            level = max(0, level - (time_passed * leak_rate))
            
            if increment:
                if level < config.limit:
                    level += 1
                    allowed = True
                else:
                    allowed = False
                
                # Update bucket state
                await self.redis_client.hset(key, mapping={
                    "level": str(level),
                    "last_leak": str(now)
                })
                await self.redis_client.expire(key, config.window)
            else:
                allowed = level < config.limit
            
            remaining = int(config.limit - level)
            reset_time = datetime.fromtimestamp(now + config.window)
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                limit=config.limit,
                window=config.window
            )
            
        except Exception as e:
            logger.error(f"Leaky bucket rate limit check failed: {e}")
            raise
    
    async def check_multiple_limits(
        self,
        identifiers: Dict[str, str],
        configs: Dict[str, RateLimitConfig],
        increment: bool = True
    ) -> Dict[str, RateLimitResult]:
        """Check multiple rate limits simultaneously"""
        try:
            tasks = []
            results = {}
            
            for limit_type, identifier in identifiers.items():
                if limit_type in configs:
                    task = self.check_rate_limit(identifier, configs[limit_type], increment)
                    tasks.append((limit_type, task))
            
            # Execute all checks in parallel
            for limit_type, task in tasks:
                try:
                    result = await task
                    results[limit_type] = result
                except Exception as e:
                    logger.error(f"Rate limit check failed for {limit_type}: {e}")
                    # Allow on error
                    results[limit_type] = RateLimitResult(
                        allowed=True,
                        remaining=configs[limit_type].limit,
                        reset_time=datetime.utcnow() + timedelta(seconds=configs[limit_type].window),
                        limit=configs[limit_type].limit,
                        window=configs[limit_type].window
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Multiple rate limit check failed: {e}")
            raise
    
    async def get_rate_limit_status(self, identifier: str, config: RateLimitConfig) -> Dict[str, Any]:
        """Get current rate limit status without incrementing"""
        try:
            result = await self.check_rate_limit(identifier, config, increment=False)
            
            return {
                "identifier": identifier,
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "window": result.window,
                "strategy": config.strategy.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {
                "identifier": identifier,
                "error": str(e)
            }
    
    async def reset_rate_limit(self, identifier: str, config: RateLimitConfig):
        """Reset rate limit for identifier"""
        try:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                now = int(time.time())
                window_start = (now // config.window) * config.window
                key = f"rate_limit:fixed:{identifier}:{window_start}"
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                key = f"rate_limit:sliding:{identifier}"
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                key = f"rate_limit:bucket:{identifier}"
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                key = f"rate_limit:leaky:{identifier}"
            else:
                raise ValueError(f"Unsupported strategy: {config.strategy}")
            
            await self.redis_client.delete(key)
            logger.info(f"Rate limit reset for {identifier}")
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            raise
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        try:
            # Get all rate limit keys
            keys = await self.redis_client.keys("rate_limit:*")
            
            stats = {
                "total_keys": len(keys),
                "strategies": {},
                "top_identifiers": []
            }
            
            # Count by strategy
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    strategy = parts[2]
                    stats["strategies"][strategy] = stats["strategies"].get(strategy, 0) + 1
            
            # Get top identifiers by usage
            identifier_usage = {}
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 4:
                    identifier = parts[3]
                    ttl = await self.redis_client.ttl(key)
                    if ttl > 0:
                        identifier_usage[identifier] = identifier_usage.get(identifier, 0) + 1
            
            # Sort by usage
            stats["top_identifiers"] = sorted(
                identifier_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get rate limit stats: {e}")
            return {"error": str(e)}
    
    def get_config_for_endpoint(self, endpoint: str) -> Optional[RateLimitConfig]:
        """Get rate limit configuration for endpoint"""
        return self.default_configs.get("endpoint", {}).get(endpoint)
    
    def get_config_for_type(self, limit_type: str) -> Optional[RateLimitConfig]:
        """Get rate limit configuration for type"""
        return self.default_configs.get(limit_type)


# Global rate limiter
rate_limiter = RateLimiter()


# Rate limiting decorators
def rate_limit(
    limit: int,
    window: int,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
    identifier_func: Optional[callable] = None
):
    """Rate limiting decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = "default"
            
            # Create config
            config = RateLimitConfig(
                strategy=strategy,
                limit=limit,
                window=window
            )
            
            # Check rate limit
            result = await rate_limiter.check_rate_limit(identifier, config)
            
            if not result.allowed:
                raise create_rate_limit_error(
                    "Rate limit exceeded",
                    details={
                        "limit": result.limit,
                        "remaining": result.remaining,
                        "reset_time": result.reset_time.isoformat(),
                        "retry_after": result.retry_after
                    }
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Rate limiting middleware
class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Get identifiers
        identifiers = {
            "global": "global",
            "ip": request.client.host if request.client else "unknown"
        }
        
        # Add user identifier if available
        if hasattr(request.state, "user_id"):
            identifiers["user"] = f"user:{request.state.user_id}"
        
        # Get configurations
        configs = {
            "global": self.rate_limiter.get_config_for_type("global"),
            "ip": self.rate_limiter.get_config_for_type("ip"),
            "user": self.rate_limiter.get_config_for_type("user"),
            "endpoint": self.rate_limiter.get_config_for_endpoint(request.url.path)
        }
        
        # Filter out None configs
        configs = {k: v for k, v in configs.items() if v is not None}
        
        # Check rate limits
        results = await self.rate_limiter.check_multiple_limits(identifiers, configs)
        
        # Check if any limit is exceeded
        for limit_type, result in results.items():
            if not result.allowed:
                # Return rate limit exceeded response
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "limit_type": limit_type,
                        "limit": result.limit,
                        "remaining": result.remaining,
                        "reset_time": result.reset_time.isoformat(),
                        "retry_after": result.retry_after
                    },
                    headers={
                        "Retry-After": str(result.retry_after) if result.retry_after else "60",
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": str(result.remaining),
                        "X-RateLimit-Reset": str(int(result.reset_time.timestamp()))
                    }
                )
                
                await response(scope, receive, send)
                return
        
        # Add rate limit headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                # Add rate limit headers
                for limit_type, result in results.items():
                    headers[f"X-RateLimit-{limit_type.title()}-Limit"] = str(result.limit)
                    headers[f"X-RateLimit-{limit_type.title()}-Remaining"] = str(result.remaining)
                    headers[f"X-RateLimit-{limit_type.title()}-Reset"] = str(int(result.reset_time.timestamp()))
                
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)





























