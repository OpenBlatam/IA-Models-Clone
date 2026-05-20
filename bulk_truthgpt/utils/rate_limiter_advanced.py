"""
Advanced Rate Limiter
=====================

Multi-strategy rate limiting with sliding window, token bucket, and leaky bucket.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    ):
        """
        Args:
            rate: Requests per second
            capacity: Maximum burst capacity (default: rate * 2)
            strategy: Rate limiting strategy
        """
        self.rate = rate
        self.capacity = capacity or (rate * 2)
        self.strategy = strategy
        
        # Token bucket
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        
        # Sliding window
        self.request_times = deque()
        
        # Fixed window
        self.window_start = time.monotonic()
        self.window_count = 0
        
        # Leaky bucket
        self.leaky_bucket = 0
        self.leak_rate = rate
        
        self.lock = asyncio.Lock()
        self.total_requests = 0
        self.blocked_requests = 0
    
    async def acquire(self, tokens: float = 1.0) -> Tuple[bool, Optional[float]]:
        """
        Try to acquire tokens.
        
        Returns:
            (allowed, wait_time_seconds)
        """
        async with self.lock:
            self.total_requests += 1
            
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._token_bucket_acquire(tokens)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._sliding_window_acquire(tokens)
            elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return self._leaky_bucket_acquire(tokens)
            else:  # FIXED_WINDOW
                return self._fixed_window_acquire(tokens)
    
    def _token_bucket_acquire(self, tokens: float) -> Tuple[bool, Optional[float]]:
        """Token bucket algorithm."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        # Refill tokens
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, None
        else:
            wait_time = (tokens - self.tokens) / self.rate
            self.blocked_requests += 1
            return False, wait_time
    
    def _sliding_window_acquire(self, tokens: float) -> Tuple[bool, Optional[float]]:
        """Sliding window algorithm."""
        now = time.monotonic()
        window_start = now - 1.0  # 1 second window
        
        # Remove old requests
        while self.request_times and self.request_times[0] < window_start:
            self.request_times.popleft()
        
        if len(self.request_times) + tokens <= self.rate:
            # Add current request
            for _ in range(int(tokens)):
                self.request_times.append(now)
            return True, None
        else:
            # Need to wait
            oldest = self.request_times[0] if self.request_times else now
            wait_time = 1.0 - (now - oldest)
            wait_time = max(0, wait_time)
            self.blocked_requests += 1
            return False, wait_time
    
    def _leaky_bucket_acquire(self, tokens: float) -> Tuple[bool, Optional[float]]:
        """Leaky bucket algorithm."""
        now = time.monotonic()
        
        # Leak bucket
        if hasattr(self, '_last_leak_time'):
            elapsed = now - self._last_leak_time
            leaked = elapsed * self.leak_rate
            self.leaky_bucket = max(0, self.leaky_bucket - leaked)
        
        self._last_leak_time = now
        
        if self.leaky_bucket + tokens <= self.capacity:
            self.leaky_bucket += tokens
            return True, None
        else:
            # Calculate wait time
            wait_time = (self.leaky_bucket + tokens - self.capacity) / self.leak_rate
            self.blocked_requests += 1
            return False, wait_time
    
    def _fixed_window_acquire(self, tokens: float) -> Tuple[bool, Optional[float]]:
        """Fixed window algorithm."""
        now = time.monotonic()
        
        # Reset window if needed
        if now - self.window_start >= 1.0:
            self.window_start = now
            self.window_count = 0
        
        if self.window_count + tokens <= self.rate:
            self.window_count += tokens
            return True, None
        else:
            # Wait until next window
            wait_time = 1.0 - (now - self.window_start)
            self.blocked_requests += 1
            return False, wait_time
    
    async def wait(self, tokens: float = 1.0):
        """Wait until tokens are available."""
        while True:
            allowed, wait_time = await self.acquire(tokens)
            if allowed:
                return
            if wait_time:
                await asyncio.sleep(wait_time)
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        total = self.total_requests
        blocked = self.blocked_requests
        allowed = total - blocked
        
        return {
            "strategy": self.strategy.value,
            "rate": self.rate,
            "capacity": self.capacity,
            "total_requests": total,
            "allowed_requests": allowed,
            "blocked_requests": blocked,
            "block_rate": round((blocked / total * 100) if total > 0 else 0, 2),
            "current_tokens": self.tokens if self.strategy == RateLimitStrategy.TOKEN_BUCKET else None,
            "current_bucket": self.leaky_bucket if self.strategy == RateLimitStrategy.LEAKY_BUCKET else None
        }
    
    def reset(self):
        """Reset rate limiter state."""
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.request_times.clear()
        self.window_start = time.monotonic()
        self.window_count = 0
        self.leaky_bucket = 0
        self.total_requests = 0
        self.blocked_requests = 0

# Global instances
default_rate_limiter = AdvancedRateLimiter(rate=10.0, capacity=50.0)
strict_rate_limiter = AdvancedRateLimiter(
    rate=5.0,
    capacity=10.0,
    strategy=RateLimitStrategy.SLIDING_WINDOW
)
































