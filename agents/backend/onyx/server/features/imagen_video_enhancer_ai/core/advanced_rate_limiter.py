"""
Advanced Rate Limiter
=====================

Advanced rate limiting with multiple strategies and distributed support.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    rate: float = 10.0  # Requests per second
    burst: int = 20  # Burst capacity
    window_seconds: float = 60.0  # For window-based strategies


class TokenBucketRateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second
            burst: Maximum bucket size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Calculate wait time
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            needed = tokens - self.tokens
            wait_time = needed / self.rate
            self.tokens = 0
            return wait_time


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, rate: float, window_seconds: float):
        """
        Initialize sliding window.
        
        Args:
            rate: Requests per window
            window_seconds: Window size in seconds
        """
        self.rate = rate
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Try to acquire permission.
        
        Returns:
            True if allowed, False otherwise
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            return False
    
    async def wait(self) -> float:
        """
        Wait until request is allowed.
        
        Returns:
            Wait time in seconds
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return 0.0
            
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = (oldest_request + self.window_seconds) - now
            return max(0.0, wait_time)


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(
        self,
        config: RateLimitConfig,
        identifier: Optional[str] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
            identifier: Optional identifier for distributed limiting
        """
        self.config = config
        self.identifier = identifier
        
        # Initialize strategy-specific limiter
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.limiter = TokenBucketRateLimiter(config.rate, config.burst)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.limiter = SlidingWindowRateLimiter(config.rate, config.window_seconds)
        else:
            raise ValueError(f"Unsupported strategy: {config.strategy}")
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire permission.
        
        Args:
            tokens: Number of tokens (for token bucket)
            
        Returns:
            True if allowed, False otherwise
        """
        if isinstance(self.limiter, TokenBucketRateLimiter):
            return await self.limiter.acquire(tokens)
        else:
            return await self.limiter.acquire()
    
    async def wait(self, tokens: int = 1) -> float:
        """
        Wait until permission is granted.
        
        Args:
            tokens: Number of tokens (for token bucket)
            
        Returns:
            Wait time in seconds
        """
        wait_time = await self.limiter.wait(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        return wait_time
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.wait()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass




