"""
Rate Limiting Implementation
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int = 100
    window_seconds: int = 60
    key_prefix: str = "rate_limit"


class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, config: RateLimitConfig, redis_client=None):
        self.config = config
        self.redis_client = redis_client
        self._local_counts: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        full_key = f"{self.config.key_prefix}:{key}"
        now = datetime.utcnow()
        
        if self.redis_client:
            return await self._check_redis(full_key, now)
        else:
            return await self._check_local(full_key, now)
    
    async def _check_redis(self, key: str, now: datetime) -> bool:
        """Check rate limit using Redis"""
        try:
            window_start = now - timedelta(seconds=self.config.window_seconds)
            window_start_ts = int(window_start.timestamp())
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start_ts)
            pipe.zcard(key)
            pipe.zadd(key, {str(now.timestamp()): now.timestamp()})
            pipe.expire(key, self.config.window_seconds)
            results = await pipe.execute()
            
            current_count = results[1]
            
            if current_count >= self.config.max_requests:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {e}")
            # Fail open - allow request if Redis fails
            return True
    
    async def _check_local(self, key: str, now: datetime) -> bool:
        """Check rate limit using local storage"""
        async with self._lock:
            if key not in self._local_counts:
                self._local_counts[key] = []
            
            # Remove old entries
            window_start = now - timedelta(seconds=self.config.window_seconds)
            self._local_counts[key] = [
                ts for ts in self._local_counts[key]
                if ts > window_start
            ]
            
            # Check count
            if len(self._local_counts[key]) >= self.config.max_requests:
                return False
            
            # Add current request
            self._local_counts[key].append(now)
            return True
    
    async def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window"""
        full_key = f"{self.config.key_prefix}:{key}"
        now = datetime.utcnow()
        
        if self.redis_client:
            window_start = now - timedelta(seconds=self.config.window_seconds)
            window_start_ts = int(window_start.timestamp())
            count = await self.redis_client.zcount(
                full_key,
                window_start_ts,
                now.timestamp()
            )
        else:
            async with self._lock:
                if full_key not in self._local_counts:
                    return self.config.max_requests
                
                window_start = now - timedelta(seconds=self.config.window_seconds)
                count = len([
                    ts for ts in self._local_counts[full_key]
                    if ts > window_start
                ])
        
        return max(0, self.config.max_requests - count)
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        full_key = f"{self.config.key_prefix}:{key}"
        
        if self.redis_client:
            await self.redis_client.delete(full_key)
        else:
            async with self._lock:
                if full_key in self._local_counts:
                    del self._local_counts[full_key]


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

