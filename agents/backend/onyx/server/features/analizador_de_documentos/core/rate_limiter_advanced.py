"""
Advanced Rate Limiter for Document Analyzer
============================================

Advanced rate limiting with multiple strategies and algorithms.
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: float
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    ):
        self.config = RateLimitConfig(
            max_requests=max_requests,
            window_seconds=window_seconds,
            strategy=strategy
        )
        
        # Token bucket
        self.tokens = max_requests
        self.last_refill = time.monotonic()
        
        # Sliding window
        self.request_times: deque = deque()
        
        # Fixed window
        self.window_start = time.monotonic()
        self.window_count = 0
        
        self.lock = asyncio.Lock()
        logger.info(f"AdvancedRateLimiter initialized: {max_requests} req/{window_seconds}s ({strategy.value})")
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens, returns True if allowed"""
        async with self.lock:
            if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._token_bucket_acquire(tokens)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._sliding_window_acquire(tokens)
            elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return self._fixed_window_acquire(tokens)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return self._leaky_bucket_acquire(tokens)
            return False
    
    def _token_bucket_acquire(self, tokens: float) -> bool:
        """Token bucket algorithm"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        
        # Refill tokens
        refill_rate = self.config.max_requests / self.config.window_seconds
        self.tokens = min(self.config.max_requests, self.tokens + elapsed * refill_rate)
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _sliding_window_acquire(self, tokens: float) -> bool:
        """Sliding window algorithm"""
        now = time.monotonic()
        cutoff = now - self.config.window_seconds
        
        # Remove old requests
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        
        if len(self.request_times) + int(tokens) <= self.config.max_requests:
            for _ in range(int(tokens)):
                self.request_times.append(now)
            return True
        return False
    
    def _fixed_window_acquire(self, tokens: float) -> bool:
        """Fixed window algorithm"""
        now = time.monotonic()
        
        # Reset window if expired
        if now - self.window_start >= self.config.window_seconds:
            self.window_start = now
            self.window_count = 0
        
        if self.window_count + int(tokens) <= self.config.max_requests:
            self.window_count += int(tokens)
            return True
        return False
    
    def _leaky_bucket_acquire(self, tokens: float) -> bool:
        """Leaky bucket algorithm (similar to token bucket)"""
        return self._token_bucket_acquire(tokens)
    
    async def wait(self, tokens: float = 1.0):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)
    
    def get_remaining(self) -> int:
        """Get remaining tokens/requests"""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return int(self.tokens)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return max(0, self.config.max_requests - len(self.request_times))
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return max(0, self.config.max_requests - self.window_count)
        return 0

# Global instance
advanced_rate_limiter = AdvancedRateLimiter()
















