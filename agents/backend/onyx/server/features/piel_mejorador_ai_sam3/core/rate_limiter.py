"""
Rate Limiter for Piel Mejorador AI SAM3
=======================================

Token bucket rate limiting implementation.
"""

import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_seconds: int = 60


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float
    tokens: float = field(default=None)
    refill_rate: float = 1.0
    last_refill: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = self.capacity
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Get time to wait before tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    
    Features:
    - Per-client rate limiting
    - Token bucket algorithm
    - Configurable limits
    - Wait time calculation
    """
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = asyncio.Lock()
        
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rate_limited_requests": 0,
        }
    
    def set_limit(
        self,
        key: str,
        config: RateLimitConfig
    ):
        """
        Set rate limit for a specific key.
        
        Args:
            key: Client identifier (IP, user ID, etc.)
            config: Rate limit configuration
        """
        self._configs[key] = config
        
        # Create or update bucket
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=config.burst_size,
                refill_rate=config.requests_per_second
            )
        else:
            bucket = self._buckets[key]
            bucket.capacity = config.burst_size
            bucket.refill_rate = config.requests_per_second
    
    async def is_allowed(
        self,
        key: str = "default",
        tokens: float = 1.0
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Client identifier
            tokens: Number of tokens to consume
            
        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            # Get config for key or use default
            config = self._configs.get(key, self.default_config)
            
            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    capacity=config.burst_size,
                    refill_rate=config.requests_per_second
                )
            
            bucket = self._buckets[key]
            
            self._stats["total_requests"] += 1
            
            if bucket.consume(tokens):
                self._stats["allowed_requests"] += 1
                return True
            else:
                self._stats["rate_limited_requests"] += 1
                return False
    
    async def get_wait_time(
        self,
        key: str = "default",
        tokens: float = 1.0
    ) -> float:
        """
        Get time to wait before request is allowed.
        
        Args:
            key: Client identifier
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait
        """
        async with self._lock:
            config = self._configs.get(key, self.default_config)
            
            if key not in self._buckets:
                return 0.0
            
            bucket = self._buckets[key]
            return bucket.get_wait_time(tokens)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        rate_limit_rate = (
            self._stats["rate_limited_requests"] / self._stats["total_requests"]
            if self._stats["total_requests"] > 0 else 0
        )
        
        return {
            **self._stats,
            "rate_limit_rate": rate_limit_rate,
            "active_buckets": len(self._buckets),
        }
    
    async def cleanup_inactive(self, max_age_seconds: int = 3600):
        """Clean up inactive buckets."""
        async with self._lock:
            now = time.time()
            inactive_keys = []
            
            for key, bucket in self._buckets.items():
                if now - bucket.last_refill > max_age_seconds:
                    inactive_keys.append(key)
            
            for key in inactive_keys:
                del self._buckets[key]
                if key in self._configs:
                    del self._configs[key]
            
            if inactive_keys:
                logger.info(f"Cleaned up {len(inactive_keys)} inactive rate limit buckets")




