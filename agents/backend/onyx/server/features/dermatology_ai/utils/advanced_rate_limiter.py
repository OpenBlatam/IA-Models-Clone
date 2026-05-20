"""
Advanced Rate Limiter with Redis Backend
Supports multiple strategies: fixed window, sliding window, token bucket
"""

import time
import json
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis[hiredis]")


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple strategies.
    Uses Redis for distributed rate limiting.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        fallback_to_memory: bool = True
    ):
        self.redis_url = redis_url
        self.strategy = strategy
        self.fallback_to_memory = fallback_to_memory
        self.client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, Any] = {}
        self.connected = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        if REDIS_AVAILABLE and self.redis_url:
            try:
                self.client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.client.ping()
                self.connected = True
                logger.info("✅ Rate limiter connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory fallback.")
                self.connected = False
        else:
            logger.info("Using in-memory rate limiter")
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
        tokens: Optional[int] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed
        
        Args:
            key: Unique identifier (user_id, IP, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            tokens: Token bucket size (for token bucket strategy)
            
        Returns:
            Tuple (allowed, info)
        """
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window(key, limit, window_seconds)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window(key, limit, window_seconds)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket(key, limit, window_seconds, tokens or limit)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    async def _fixed_window(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""
        window = int(time.time()) // window_seconds
        window_key = f"{key}:{window}"
        
        if self.connected and self.client:
            current = await self.client.incr(window_key)
            if current == 1:
                await self.client.expire(window_key, window_seconds)
        else:
            # Memory fallback
            if window_key not in self.memory_store:
                self.memory_store[window_key] = 0
            self.memory_store[window_key] += 1
            current = self.memory_store[window_key]
        
        allowed = current <= limit
        remaining = max(0, limit - current)
        
        return allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset_at": (window + 1) * window_seconds
        }
    
    async def _sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting"""
        now = time.time()
        window_start = now - window_seconds
        
        if self.connected and self.client:
            # Use sorted set for sliding window
            zset_key = f"{key}:sliding"
            
            # Remove old entries
            await self.client.zremrangebyscore(zset_key, 0, window_start)
            
            # Count current requests
            current = await self.client.zcard(zset_key)
            
            if current < limit:
                # Add new request
                await self.client.zadd(zset_key, {str(now): now})
                await self.client.expire(zset_key, window_seconds)
                current += 1
                allowed = True
            else:
                allowed = False
            
            # Get oldest request for reset time
            oldest = await self.client.zrange(zset_key, 0, 0, withscores=True)
            reset_at = oldest[0][1] + window_seconds if oldest else now + window_seconds
            
        else:
            # Memory fallback
            if key not in self.memory_store:
                self.memory_store[key] = []
            
            # Remove old entries
            self.memory_store[key] = [
                ts for ts in self.memory_store[key]
                if ts > window_start
            ]
            
            current = len(self.memory_store[key])
            
            if current < limit:
                self.memory_store[key].append(now)
                current += 1
                allowed = True
            else:
                allowed = False
            
            reset_at = (min(self.memory_store[key]) if self.memory_store[key] else now) + window_seconds
        
        remaining = max(0, limit - current)
        
        return allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset_at": int(reset_at)
        }
    
    async def _token_bucket(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""
        now = time.time()
        bucket_key = f"{key}:bucket"
        last_refill_key = f"{key}:last_refill"
        
        if self.connected and self.client:
            # Get current state
            pipe = self.client.pipeline()
            pipe.get(bucket_key)
            pipe.get(last_refill_key)
            results = await pipe.execute()
            
            current_tokens = int(results[0]) if results[0] else tokens
            last_refill = float(results[1]) if results[1] else now
            
            # Refill tokens
            time_passed = now - last_refill
            tokens_to_add = int((time_passed / window_seconds) * limit)
            
            if tokens_to_add > 0:
                current_tokens = min(tokens, current_tokens + tokens_to_add)
                await self.client.set(bucket_key, current_tokens)
                await self.client.set(last_refill_key, now)
                await self.client.expire(bucket_key, window_seconds * 2)
                await self.client.expire(last_refill_key, window_seconds * 2)
            
            # Check if request is allowed
            if current_tokens > 0:
                current_tokens -= 1
                await self.client.set(bucket_key, current_tokens)
                allowed = True
            else:
                allowed = False
            
        else:
            # Memory fallback
            if bucket_key not in self.memory_store:
                self.memory_store[bucket_key] = tokens
                self.memory_store[last_refill_key] = now
            
            current_tokens = self.memory_store[bucket_key]
            last_refill = self.memory_store[last_refill_key]
            
            # Refill tokens
            time_passed = now - last_refill
            tokens_to_add = int((time_passed / window_seconds) * limit)
            
            if tokens_to_add > 0:
                current_tokens = min(tokens, current_tokens + tokens_to_add)
                self.memory_store[bucket_key] = current_tokens
                self.memory_store[last_refill_key] = now
            
            # Check if request is allowed
            if current_tokens > 0:
                self.memory_store[bucket_key] -= 1
                allowed = True
            else:
                allowed = False
        
        remaining = max(0, current_tokens - 1) if allowed else 0
        reset_at = now + window_seconds
        
        return allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset_at": int(reset_at),
            "tokens": current_tokens
        }
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        if self.connected and self.client:
            # Delete all keys related to this identifier
            pattern = f"{key}:*"
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)
        else:
            # Memory fallback
            keys_to_delete = [k for k in self.memory_store.keys() if k.startswith(f"{key}:")]
            for k in keys_to_delete:
                del self.memory_store[k]
    
    async def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limit statistics for a key"""
        # Implementation depends on strategy
        return {
            "key": key,
            "strategy": self.strategy.value,
            "connected": self.connected
        }
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.connected = False















