from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aioredis
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
Advanced Rate Limiter
=====================

Advanced rate limiting system with multiple algorithms, distributed rate limiting,
and intelligent throttling for LinkedIn posts system.
"""




logger = get_logger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    window_size: int = 60  # seconds
    retry_after_header: bool = True
    enable_adaptive: bool = True


@dataclass
class RateLimitInfo:
    """Rate limit information."""
    limit: int
    remaining: int
    reset_time: int
    retry_after: int
    window_start: int
    window_end: int


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple algorithms and distributed support.
    
    Features:
    - Multiple rate limiting algorithms
    - Distributed rate limiting with Redis
    - Adaptive rate limiting
    - Burst handling
    - Rate limit analytics
    - Custom rate limit rules
    - Rate limit bypass for VIP users
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_config: Optional[RateLimitConfig] = None,
        enable_distributed: bool = True,
        enable_analytics: bool = True,
    ):
        """Initialize the advanced rate limiter."""
        self.redis_url = redis_url
        self.enable_distributed = enable_distributed
        self.enable_analytics = enable_analytics
        
        # Default configuration
        self.default_config = default_config or RateLimitConfig()
        
        # Redis client
        self.redis_client = None
        self.redis_pool = None
        
        # Local rate limit storage (fallback)
        self.local_limits = defaultdict(lambda: {
            "requests": deque(),
            "last_reset": time.time(),
            "current_count": 0
        })
        
        # Rate limit analytics
        self.analytics = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "bypass_requests": 0,
            "algorithm_usage": defaultdict(int),
            "user_limits": defaultdict(int),
        }
        
        # Custom rate limit rules
        self.custom_rules = {}
        
        # VIP users (bypass rate limits)
        self.vip_users = set()
        
        # Initialize Redis connection
        if enable_distributed:
            asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=False
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis for rate limiting: {e}")
            self.redis_client = None
            self.enable_distributed = False
    
    def _generate_key(self, identifier: str, window: str = "minute") -> str:
        """Generate Redis key for rate limiting."""
        timestamp = int(time.time())
        if window == "minute":
            window_timestamp = timestamp - (timestamp % 60)
        elif window == "hour":
            window_timestamp = timestamp - (timestamp % 3600)
        elif window == "day":
            window_timestamp = timestamp - (timestamp % 86400)
        else:
            window_timestamp = timestamp
        
        return f"rate_limit:{identifier}:{window}:{window_timestamp}"
    
    def _get_user_identifier(self, user_id: Optional[str] = None, ip_address: Optional[str] = None) -> str:
        """Get user identifier for rate limiting."""
        if user_id:
            return f"user:{user_id}"
        elif ip_address:
            return f"ip:{ip_address}"
        else:
            return "anonymous"
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: Optional[RateLimitConfig] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        bypass_vip: bool = True
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        try:
            # Check if user is VIP
            if bypass_vip and user_id and user_id in self.vip_users:
                self.analytics["bypass_requests"] += 1
                return True, RateLimitInfo(
                    limit=0,
                    remaining=0,
                    reset_time=0,
                    retry_after=0,
                    window_start=0,
                    window_end=0
                )
            
            # Use provided config or default
            rate_config = config or self.default_config
            
            # Check custom rules
            custom_rule = self.custom_rules.get(identifier)
            if custom_rule:
                rate_config = custom_rule
            
            # Get user identifier
            user_identifier = self._get_user_identifier(user_id, ip_address)
            full_identifier = f"{identifier}:{user_identifier}"
            
            # Update analytics
            self.analytics["total_requests"] += 1
            self.analytics["algorithm_usage"][rate_config.algorithm.value] += 1
            
            # Check rate limit based on algorithm
            if rate_config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                allowed, info = await self._check_fixed_window(full_identifier, rate_config)
            elif rate_config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                allowed, info = await self._check_sliding_window(full_identifier, rate_config)
            elif rate_config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                allowed, info = await self._check_leaky_bucket(full_identifier, rate_config)
            elif rate_config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                allowed, info = await self._check_token_bucket(full_identifier, rate_config)
            elif rate_config.algorithm == RateLimitAlgorithm.ADAPTIVE:
                allowed, info = await self._check_adaptive(full_identifier, rate_config)
            else:
                allowed, info = await self._check_sliding_window(full_identifier, rate_config)
            
            # Update analytics
            if not allowed:
                self.analytics["rate_limited_requests"] += 1
                self.analytics["user_limits"][user_identifier] += 1
            
            return allowed, info
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Allow request on error (fail open)
            return True, RateLimitInfo(
                limit=0,
                remaining=0,
                reset_time=0,
                retry_after=0,
                window_start=0,
                window_end=0
            )
    
    async def _check_fixed_window(self, identifier: str, config: RateLimitConfig) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using fixed window algorithm."""
        current_time = int(time.time())
        window_start = current_time - (current_time % config.window_size)
        window_end = window_start + config.window_size
        
        if self.enable_distributed and self.redis_client:
            # Use Redis for distributed rate limiting
            key = self._generate_key(identifier, "minute")
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, config.window_size)
            results = await pipe.execute()
            
            current_count = results[0]
            limit = config.requests_per_minute
            
        else:
            # Use local storage
            local_data = self.local_limits[identifier]
            
            # Reset if window has passed
            if current_time >= local_data["last_reset"] + config.window_size:
                local_data["current_count"] = 0
                local_data["last_reset"] = current_time
            
            local_data["current_count"] += 1
            current_count = local_data["current_count"]
            limit = config.requests_per_minute
        
        allowed = current_count <= limit
        remaining = max(0, limit - current_count)
        retry_after = window_end - current_time if not allowed else 0
        
        return allowed, RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=window_end,
            retry_after=retry_after,
            window_start=window_start,
            window_end=window_end
        )
    
    async def _check_sliding_window(self, identifier: str, config: RateLimitConfig) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using sliding window algorithm."""
        current_time = int(time.time())
        window_start = current_time - config.window_size
        window_end = current_time
        
        if self.enable_distributed and self.redis_client:
            # Use Redis sorted set for sliding window
            key = f"rate_limit:sliding:{identifier}"
            
            # Add current request
            await self.redis_client.zadd(key, {str(current_time): current_time})
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            current_count = await self.redis_client.zcard(key)
            
            # Set expiration
            await self.redis_client.expire(key, config.window_size)
            
        else:
            # Use local storage with sliding window
            local_data = self.local_limits[identifier]
            
            # Add current request
            local_data["requests"].append(current_time)
            
            # Remove old requests
            while local_data["requests"] and local_data["requests"][0] < window_start:
                local_data["requests"].popleft()
            
            current_count = len(local_data["requests"])
        
        limit = config.requests_per_minute
        allowed = current_count <= limit
        remaining = max(0, limit - current_count)
        retry_after = config.window_size if not allowed else 0
        
        return allowed, RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=window_end + config.window_size,
            retry_after=retry_after,
            window_start=window_start,
            window_end=window_end
        )
    
    async def _check_leaky_bucket(self, identifier: str, config: RateLimitConfig) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using leaky bucket algorithm."""
        current_time = int(time.time())
        
        if self.enable_distributed and self.redis_client:
            # Use Redis for leaky bucket
            bucket_key = f"rate_limit:leaky:{identifier}"
            
            # Get current bucket state
            bucket_data = await self.redis_client.get(bucket_key)
            if bucket_data:
                bucket_info = json.loads(bucket_data)
                last_leak_time = bucket_info["last_leak_time"]
                current_tokens = bucket_info["tokens"]
            else:
                last_leak_time = current_time
                current_tokens = config.burst_limit
            
            # Calculate leaked tokens
            time_passed = current_time - last_leak_time
            leak_rate = config.requests_per_minute / 60  # tokens per second
            leaked_tokens = time_passed * leak_rate
            current_tokens = min(config.burst_limit, current_tokens + leaked_tokens)
            
            # Check if request can be processed
            if current_tokens >= 1:
                current_tokens -= 1
                allowed = True
            else:
                allowed = False
            
            # Update bucket state
            bucket_info = {
                "last_leak_time": current_time,
                "tokens": current_tokens
            }
            await self.redis_client.setex(bucket_key, config.window_size, json.dumps(bucket_info))
            
        else:
            # Simplified local implementation
            allowed = True  # Always allow for local implementation
            current_tokens = config.burst_limit
        
        limit = config.burst_limit
        remaining = int(current_tokens)
        retry_after = int((1 - current_tokens) / (config.requests_per_minute / 60)) if not allowed else 0
        
        return allowed, RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=current_time + config.window_size,
            retry_after=retry_after,
            window_start=current_time,
            window_end=current_time + config.window_size
        )
    
    async def _check_token_bucket(self, identifier: str, config: RateLimitConfig) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using token bucket algorithm."""
        current_time = int(time.time())
        
        if self.enable_distributed and self.redis_client:
            # Use Redis for token bucket
            bucket_key = f"rate_limit:token:{identifier}"
            
            # Get current bucket state
            bucket_data = await self.redis_client.get(bucket_key)
            if bucket_data:
                bucket_info = json.loads(bucket_data)
                last_refill_time = bucket_info["last_refill_time"]
                current_tokens = bucket_info["tokens"]
            else:
                last_refill_time = current_time
                current_tokens = config.burst_limit
            
            # Calculate refilled tokens
            time_passed = current_time - last_refill_time
            refill_rate = config.requests_per_minute / 60  # tokens per second
            refilled_tokens = time_passed * refill_rate
            current_tokens = min(config.burst_limit, current_tokens + refilled_tokens)
            
            # Check if request can be processed
            if current_tokens >= 1:
                current_tokens -= 1
                allowed = True
            else:
                allowed = False
            
            # Update bucket state
            bucket_info = {
                "last_refill_time": current_time,
                "tokens": current_tokens
            }
            await self.redis_client.setex(bucket_key, config.window_size, json.dumps(bucket_info))
            
        else:
            # Simplified local implementation
            allowed = True  # Always allow for local implementation
            current_tokens = config.burst_limit
        
        limit = config.burst_limit
        remaining = int(current_tokens)
        retry_after = int((1 - current_tokens) / (config.requests_per_minute / 60)) if not allowed else 0
        
        return allowed, RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=current_time + config.window_size,
            retry_after=retry_after,
            window_start=current_time,
            window_end=current_time + config.window_size
        )
    
    async def _check_adaptive(self, identifier: str, config: RateLimitConfig) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using adaptive algorithm."""
        # Adaptive rate limiting based on system load and user behavior
        current_time = int(time.time())
        
        # Get system load (mock implementation)
        system_load = await self._get_system_load()
        
        # Adjust limits based on load
        if system_load > 0.8:  # High load
            adjusted_limit = int(config.requests_per_minute * 0.5)
        elif system_load > 0.6:  # Medium load
            adjusted_limit = int(config.requests_per_minute * 0.8)
        else:  # Low load
            adjusted_limit = config.requests_per_minute
        
        # Check user behavior
        user_behavior_score = await self._get_user_behavior_score(identifier)
        if user_behavior_score < 0.5:  # Suspicious behavior
            adjusted_limit = int(adjusted_limit * 0.3)
        
        # Use sliding window with adjusted limit
        adjusted_config = RateLimitConfig(
            requests_per_minute=adjusted_limit,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            window_size=config.window_size
        )
        
        return await self._check_sliding_window(identifier, adjusted_config)
    
    async def _get_system_load(self) -> float:
        """Get current system load (mock implementation)."""
        # In a real implementation, this would check CPU, memory, etc.
        return 0.3  # Mock load of 30%
    
    async def _get_user_behavior_score(self, identifier: str) -> float:
        """Get user behavior score (mock implementation)."""
        # In a real implementation, this would analyze user behavior patterns
        return 0.8  # Mock score of 80%
    
    def add_custom_rule(self, identifier: str, config: RateLimitConfig):
        """Add custom rate limit rule."""
        self.custom_rules[identifier] = config
        logger.info(f"Added custom rate limit rule for {identifier}")
    
    def remove_custom_rule(self, identifier: str):
        """Remove custom rate limit rule."""
        if identifier in self.custom_rules:
            del self.custom_rules[identifier]
            logger.info(f"Removed custom rate limit rule for {identifier}")
    
    def add_vip_user(self, user_id: str):
        """Add VIP user (bypass rate limits)."""
        self.vip_users.add(user_id)
        logger.info(f"Added VIP user: {user_id}")
    
    def remove_vip_user(self, user_id: str):
        """Remove VIP user."""
        if user_id in self.vip_users:
            self.vip_users.remove(user_id)
            logger.info(f"Removed VIP user: {user_id}")
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get rate limiting analytics."""
        return {
            **self.analytics,
            "custom_rules_count": len(self.custom_rules),
            "vip_users_count": len(self.vip_users),
            "distributed_enabled": self.enable_distributed,
            "redis_connected": self.redis_client is not None,
        }
    
    async def reset_limits(self, identifier: str):
        """Reset rate limits for an identifier."""
        try:
            if self.enable_distributed and self.redis_client:
                # Delete all rate limit keys for this identifier
                pattern = f"rate_limit:*:{identifier}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            # Reset local limits
            if identifier in self.local_limits:
                del self.local_limits[identifier]
            
            logger.info(f"Reset rate limits for {identifier}")
            
        except Exception as e:
            logger.error(f"Error resetting rate limits: {e}")
    
    async def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for an identifier."""
        try:
            current_time = int(time.time())
            
            if self.enable_distributed and self.redis_client:
                # Get Redis keys for this identifier
                pattern = f"rate_limit:*:{identifier}:*"
                keys = await self.redis_client.keys(pattern)
                
                status = {
                    "identifier": identifier,
                    "distributed": True,
                    "redis_keys": len(keys),
                    "current_time": current_time,
                }
                
                # Get current counts for different windows
                for window in ["minute", "hour", "day"]:
                    key = self._generate_key(identifier, window)
                    count = await self.redis_client.get(key)
                    status[f"{window}_count"] = int(count) if count else 0
                
            else:
                # Local status
                local_data = self.local_limits.get(identifier, {})
                status = {
                    "identifier": identifier,
                    "distributed": False,
                    "current_count": local_data.get("current_count", 0),
                    "last_reset": local_data.get("last_reset", 0),
                    "current_time": current_time,
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            return {"error": str(e)}


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()


def get_rate_limiter() -> AdvancedRateLimiter:
    """Get global rate limiter instance."""
    return rate_limiter


def rate_limit_decorator(
    requests_per_minute: int = 60,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
    user_id_func: Optional[Callable] = None,
    ip_func: Optional[Callable] = None
):
    """Decorator for rate limiting functions."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get user identifier
            user_id = None
            ip_address = None
            
            if user_id_func:
                user_id = user_id_func(*args, **kwargs)
            if ip_func:
                ip_address = ip_func(*args, **kwargs)
            
            # Check rate limit
            config = RateLimitConfig(
                requests_per_minute=requests_per_minute,
                algorithm=algorithm
            )
            
            allowed, info = await rate_limiter.check_rate_limit(
                identifier=func.__name__,
                config=config,
                user_id=user_id,
                ip_address=ip_address
            )
            
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {info.retry_after} seconds",
                    retry_after=info.retry_after
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator 