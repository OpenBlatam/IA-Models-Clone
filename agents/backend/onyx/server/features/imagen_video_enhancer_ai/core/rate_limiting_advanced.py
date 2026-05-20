"""
Advanced Rate Limiting System
==============================

Advanced rate limiting system with multiple strategies and per-user limits.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

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
    requests_per_period: int
    period_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    burst_size: Optional[int] = None
    per_user: bool = False
    per_endpoint: bool = False


@dataclass
class RateLimitResult:
    """Rate limit result."""
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: Optional[float] = None


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize advanced rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.windows: Dict[str, List[float]] = defaultdict(list)
        self.tokens: Dict[str, float] = defaultdict(float)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: User/client identifier
            endpoint: Optional endpoint path
            
        Returns:
            Rate limit result
        """
        async with self.lock:
            # Build key
            key = identifier
            if self.config.per_endpoint and endpoint:
                key = f"{identifier}:{endpoint}"
            
            now = time.time()
            
            if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, now)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, now)
            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, now)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._check_leaky_bucket(key, now)
            else:
                # Default to fixed window
                return await self._check_fixed_window(key, now)
    
    async def _check_fixed_window(self, key: str, now: float) -> RateLimitResult:
        """Check fixed window rate limit."""
        window_start = now - (now % self.config.period_seconds)
        
        # Clean old windows
        if key in self.windows:
            self.windows[key] = [
                req_time for req_time in self.windows[key]
                if req_time >= window_start
            ]
        else:
            self.windows[key] = []
        
        # Check limit
        if len(self.windows[key]) >= self.config.requests_per_period:
            reset_at = datetime.fromtimestamp(window_start + self.config.period_seconds)
            retry_after = (window_start + self.config.period_seconds) - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Allow request
        self.windows[key].append(now)
        remaining = self.config.requests_per_period - len(self.windows[key])
        reset_at = datetime.fromtimestamp(window_start + self.config.period_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at
        )
    
    async def _check_sliding_window(self, key: str, now: float) -> RateLimitResult:
        """Check sliding window rate limit."""
        window_start = now - self.config.period_seconds
        
        # Clean old requests
        if key in self.windows:
            self.windows[key] = [
                req_time for req_time in self.windows[key]
                if req_time >= window_start
            ]
        else:
            self.windows[key] = []
        
        # Check limit
        if len(self.windows[key]) >= self.config.requests_per_period:
            oldest_request = min(self.windows[key])
            reset_at = datetime.fromtimestamp(oldest_request + self.config.period_seconds)
            retry_after = (oldest_request + self.config.period_seconds) - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Allow request
        self.windows[key].append(now)
        remaining = self.config.requests_per_period - len(self.windows[key])
        reset_at = datetime.fromtimestamp(now + self.config.period_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at
        )
    
    async def _check_token_bucket(self, key: str, now: float) -> RateLimitResult:
        """Check token bucket rate limit."""
        # Calculate tokens to add
        time_passed = now - self.last_update[key]
        tokens_to_add = (time_passed / self.config.period_seconds) * self.config.requests_per_period
        
        # Update tokens
        current_tokens = self.tokens[key] + tokens_to_add
        max_tokens = self.config.burst_size or self.config.requests_per_period
        current_tokens = min(current_tokens, max_tokens)
        
        self.tokens[key] = current_tokens
        self.last_update[key] = now
        
        # Check if we have tokens
        if current_tokens < 1:
            # Calculate when next token will be available
            tokens_needed = 1 - current_tokens
            time_needed = (tokens_needed / self.config.requests_per_period) * self.config.period_seconds
            reset_at = datetime.fromtimestamp(now + time_needed)
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=time_needed
            )
        
        # Consume token
        self.tokens[key] -= 1
        remaining = int(current_tokens - 1)
        reset_at = datetime.fromtimestamp(now + self.config.period_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at
        )
    
    async def _check_leaky_bucket(self, key: str, now: float) -> RateLimitResult:
        """Check leaky bucket rate limit."""
        # Similar to token bucket but with different semantics
        # For simplicity, using similar implementation
        return await self._check_token_bucket(key, now)
    
    async def reset(self, identifier: str, endpoint: Optional[str] = None):
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: User/client identifier
            endpoint: Optional endpoint path
        """
        async with self.lock:
            key = identifier
            if self.config.per_endpoint and endpoint:
                key = f"{identifier}:{endpoint}"
            
            if key in self.windows:
                del self.windows[key]
            if key in self.tokens:
                del self.tokens[key]
            if key in self.last_update:
                del self.last_update[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "strategy": self.config.strategy.value,
            "requests_per_period": self.config.requests_per_period,
            "period_seconds": self.config.period_seconds,
            "active_windows": len(self.windows),
            "active_tokens": len(self.tokens)
        }



