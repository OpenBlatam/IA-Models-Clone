from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
from typing import Optional
import redis.asyncio as redis
from ...core.interfaces.rate_limit_interface import IRateLimitService
from ...core.entities.rate_limit import RateLimitInfo
from ...shared.constants import RATE_LIMIT_KEY_PREFIX
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Rate Limit Implementation
===============================

Concrete implementation of rate limiting using Redis.
"""


logger = logging.getLogger(__name__)


class RedisRateLimitService(IRateLimitService):
    """Redis-based rate limiting service."""
    
    def __init__(self, redis_url: str, requests_per_window: int = 1000, window_size: int = 3600):
        
    """__init__ function."""
self.redis_url = redis_url
        self.requests_per_window = requests_per_window
        self.window_size = window_size
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            await self.redis_client.ping()
            logger.info("Redis rate limiter initialized successfully")
        except Exception as e:
            logger.warning(f"Redis rate limiter initialization failed: {e}")
            self.redis_client = None
    
    async def is_allowed(self, identifier: str) -> RateLimitInfo:
        """Check if request is allowed under rate limit."""
        if not self.redis_client:
            return RateLimitInfo.create_inactive()
        
        key = f"{RATE_LIMIT_KEY_PREFIX}{identifier}"
        current_time = time.time()
        window_start = current_time - self.window_size
        
        try:
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcount(key, window_start, current_time)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, self.window_size)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            allowed = current_requests < self.requests_per_window
            remaining = max(0, self.requests_per_window - current_requests - 1)
            
            if allowed:
                return RateLimitInfo.create_allowed(
                    remaining=remaining,
                    window_size=self.window_size,
                    current=current_requests + 1
                )
            else:
                return RateLimitInfo.create_denied(
                    window_size=self.window_size,
                    current=current_requests,
                    retry_after=self.window_size
                )
                
        except Exception as e:
            logger.warning(f"Rate limiting failed for {identifier}: {e}")
            return RateLimitInfo.create_inactive()
    
    async def reset_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier."""
        if not self.redis_client:
            return False
        
        key = f"{RATE_LIMIT_KEY_PREFIX}{identifier}"
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Failed to reset rate limit for {identifier}: {e}")
            return False
    
    async async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        if not self.redis_client:
            return self.requests_per_window
        
        key = f"{RATE_LIMIT_KEY_PREFIX}{identifier}"
        current_time = time.time()
        window_start = current_time - self.window_size
        
        try:
            current_requests = await self.redis_client.zcount(key, window_start, current_time)
            return max(0, self.requests_per_window - current_requests)
        except Exception as e:
            logger.warning(f"Failed to get remaining requests for {identifier}: {e}")
            return self.requests_per_window
    
    async def get_window_info(self, identifier: str) -> dict:
        """Get window information for identifier."""
        if not self.redis_client:
            return {
                "window_size": self.window_size,
                "requests_per_window": self.requests_per_window,
                "current_requests": 0,
                "remaining_requests": self.requests_per_window,
                "redis_available": False
            }
        
        key = f"{RATE_LIMIT_KEY_PREFIX}{identifier}"
        current_time = time.time()
        window_start = current_time - self.window_size
        
        try:
            current_requests = await self.redis_client.zcount(key, window_start, current_time)
            remaining = max(0, self.requests_per_window - current_requests)
            
            return {
                "window_size": self.window_size,
                "requests_per_window": self.requests_per_window,
                "current_requests": current_requests,
                "remaining_requests": remaining,
                "window_start": window_start,
                "current_time": current_time,
                "redis_available": True
            }
        except Exception as e:
            logger.warning(f"Failed to get window info for {identifier}: {e}")
            return {
                "window_size": self.window_size,
                "requests_per_window": self.requests_per_window,
                "current_requests": 0,
                "remaining_requests": self.requests_per_window,
                "redis_available": False,
                "error": str(e)
            } 