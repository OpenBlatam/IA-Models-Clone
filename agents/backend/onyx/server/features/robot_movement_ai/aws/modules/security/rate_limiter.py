"""
Advanced Rate Limiter
=====================

Advanced rate limiting with multiple strategies.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    limit: int = 100
    window: int = 60  # seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    per_user: bool = False
    per_ip: bool = True
    per_endpoint: bool = False


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, cache: CachePort):
        self.cache = cache
        self._configs: Dict[str, RateLimitConfig] = {}
    
    def configure(self, key: str, config: RateLimitConfig):
        """Configure rate limit for a key."""
        self._configs[key] = config
        logger.info(f"Configured rate limit for {key}: {config.limit}/{config.window}s")
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
        config: Optional[RateLimitConfig] = None
    ) -> Tuple[bool, int, int]:
        """
        Check rate limit.
        
        Returns:
            (allowed, remaining, reset_after)
        """
        if not config:
            # Get default or endpoint-specific config
            config_key = endpoint or "default"
            config = self._configs.get(config_key, RateLimitConfig())
        
        # Build cache key
        cache_key = self._build_cache_key(identifier, endpoint, config)
        
        if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window(cache_key, config)
        elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window(cache_key, config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket(cache_key, config)
        else:
            return await self._sliding_window(cache_key, config)
    
    def _build_cache_key(self, identifier: str, endpoint: Optional[str], config: RateLimitConfig) -> str:
        """Build cache key for rate limiting."""
        parts = ["rate_limit"]
        
        if config.per_ip:
            parts.append(f"ip:{identifier}")
        if config.per_user:
            parts.append(f"user:{identifier}")
        if config.per_endpoint and endpoint:
            parts.append(f"endpoint:{endpoint}")
        
        parts.append(f"{config.limit}:{config.window}")
        return ":".join(parts)
    
    async def _sliding_window(self, key: str, config: RateLimitConfig) -> Tuple[bool, int, int]:
        """Sliding window rate limiting."""
        now = time.time()
        window_start = now - config.window
        
        # Get current count
        current = await self.cache.get(key) or []
        
        # Filter out old entries
        current = [ts for ts in current if ts > window_start]
        
        # Check limit
        if len(current) >= config.limit:
            oldest = min(current) if current else now
            reset_after = int(oldest + config.window - now)
            return False, 0, reset_after
        
        # Add current request
        current.append(now)
        await self.cache.set(key, current, ttl=config.window * 2)
        
        remaining = config.limit - len(current)
        return True, remaining, config.window
    
    async def _fixed_window(self, key: str, config: RateLimitConfig) -> Tuple[bool, int, int]:
        """Fixed window rate limiting."""
        window = int(time.time() // config.window)
        window_key = f"{key}:{window}"
        
        current = await self.cache.get(window_key) or 0
        
        if current >= config.limit:
            reset_after = config.window - (int(time.time()) % config.window)
            return False, 0, reset_after
        
        # Increment
        new_count = current + 1
        await self.cache.set(window_key, new_count, ttl=config.window)
        
        remaining = config.limit - new_count
        return True, remaining, config.window - (int(time.time()) % config.window)
    
    async def _token_bucket(self, key: str, config: RateLimitConfig) -> Tuple[bool, int, int]:
        """Token bucket rate limiting."""
        bucket_key = f"{key}:bucket"
        last_refill_key = f"{key}:last_refill"
        
        now = time.time()
        last_refill = await self.cache.get(last_refill_key) or now
        tokens = await self.cache.get(bucket_key) or config.limit
        
        # Refill tokens
        elapsed = now - last_refill
        refill_rate = config.limit / config.window
        tokens = min(config.limit, tokens + elapsed * refill_rate)
        
        if tokens < 1:
            reset_after = int((1 - tokens) / refill_rate)
            return False, 0, reset_after
        
        # Consume token
        tokens -= 1
        await self.cache.set(bucket_key, tokens, ttl=config.window * 2)
        await self.cache.set(last_refill_key, now, ttl=config.window * 2)
        
        remaining = int(tokens)
        return True, remaining, int(config.window)















