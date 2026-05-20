"""
Rate limiting system for KV cache.

This module provides rate limiting capabilities to prevent cache abuse
and ensure fair resource usage.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"  # Fixed time window
    SLIDING_WINDOW = "sliding_window"  # Sliding time window
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"  # Leaky bucket algorithm


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    strategy: RateLimitStrategy
    max_requests: int
    window_seconds: float
    burst_size: Optional[int] = None  # For token bucket
    refill_rate: Optional[float] = None  # Tokens per second


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None


class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._request_times: Dict[str, deque] = defaultdict(lambda: deque())
        self._tokens: Dict[str, float] = defaultdict(lambda: config.max_requests)
        self._last_refill: Dict[str, float] = defaultdict(time.time)
        self._lock = threading.Lock()
        
    def check(self, identifier: str) -> RateLimitResult:
        """Check if request is allowed."""
        current_time = time.time()
        
        with self._lock:
            if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return self._check_fixed_window(identifier, current_time)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._check_sliding_window(identifier, current_time)
            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._check_token_bucket(identifier, current_time)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return self._check_leaky_bucket(identifier, current_time)
            else:
                return RateLimitResult(
                    allowed=True,
                    remaining=self.config.max_requests,
                    reset_time=current_time + self.config.window_seconds
                )
                
    def _check_fixed_window(self, identifier: str, current_time: float) -> RateLimitResult:
        """Fixed window rate limiting."""
        window_start = current_time - self.config.window_seconds
        
        # Clean old requests
        request_times = self._request_times[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
            
        # Check limit
        if len(request_times) < self.config.max_requests:
            request_times.append(current_time)
            remaining = self.config.max_requests - len(request_times)
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=window_start + self.config.window_seconds * 2
            )
        else:
            retry_after = request_times[0] + self.config.window_seconds - current_time
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=window_start + self.config.window_seconds * 2,
                retry_after=max(0, retry_after)
            )
            
    def _check_sliding_window(self, identifier: str, current_time: float) -> RateLimitResult:
        """Sliding window rate limiting."""
        window_start = current_time - self.config.window_seconds
        
        # Clean old requests
        request_times = self._request_times[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
            
        # Check limit
        if len(request_times) < self.config.max_requests:
            request_times.append(current_time)
            remaining = self.config.max_requests - len(request_times)
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=current_time + self.config.window_seconds
            )
        else:
            retry_after = request_times[0] + self.config.window_seconds - current_time
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=current_time + self.config.window_seconds,
                retry_after=max(0, retry_after)
            )
            
    def _check_token_bucket(self, identifier: str, current_time: float) -> RateLimitResult:
        """Token bucket rate limiting."""
        # Refill tokens
        last_refill = self._last_refill[identifier]
        time_passed = current_time - last_refill
        
        if self.config.refill_rate:
            tokens_to_add = time_passed * self.config.refill_rate
            self._tokens[identifier] = min(
                self.config.burst_size or self.config.max_requests,
                self._tokens[identifier] + tokens_to_add
            )
            self._last_refill[identifier] = current_time
            
        # Check if we have tokens
        if self._tokens[identifier] >= 1.0:
            self._tokens[identifier] -= 1.0
            remaining = int(self._tokens[identifier])
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=current_time + (1.0 / self.config.refill_rate) if self.config.refill_rate else current_time + 1.0
            )
        else:
            # Calculate retry after
            tokens_needed = 1.0 - self._tokens[identifier]
            retry_after = tokens_needed / self.config.refill_rate if self.config.refill_rate else 1.0
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
    def _check_leaky_bucket(self, identifier: str, current_time: float) -> RateLimitResult:
        """Leaky bucket rate limiting."""
        # Similar to token bucket but with different semantics
        # For simplicity, use sliding window approach
        return self._check_sliding_window(identifier, current_time)


class CacheRateLimiter:
    """Rate limiter for cache operations."""
    
    def __init__(
        self,
        cache: Any,
        get_config: Optional[RateLimitConfig] = None,
        put_config: Optional[RateLimitConfig] = None,
        delete_config: Optional[RateLimitConfig] = None
    ):
        self.cache = cache
        
        # Default configs
        default_config = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests=100,
            window_seconds=60.0
        )
        
        self.get_limiter = RateLimiter(get_config or default_config)
        self.put_limiter = RateLimiter(put_config or default_config)
        self.delete_limiter = RateLimiter(delete_config or default_config)
        
    def get(self, key: str, identifier: Optional[str] = None) -> Tuple[Any, RateLimitResult]:
        """Get with rate limiting."""
        id = identifier or "default"
        result = self.get_limiter.check(id)
        
        if result.allowed:
            value = self.cache.get(key)
            return value, result
        else:
            raise RateLimitExceededError(
                f"Rate limit exceeded for get operation. Retry after {result.retry_after:.2f} seconds"
            )
            
    def put(self, key: str, value: Any, identifier: Optional[str] = None) -> Tuple[bool, RateLimitResult]:
        """Put with rate limiting."""
        id = identifier or "default"
        result = self.put_limiter.check(id)
        
        if result.allowed:
            success = self.cache.put(key, value)
            return success, result
        else:
            raise RateLimitExceededError(
                f"Rate limit exceeded for put operation. Retry after {result.retry_after:.2f} seconds"
            )
            
    def delete(self, key: str, identifier: Optional[str] = None) -> Tuple[bool, RateLimitResult]:
        """Delete with rate limiting."""
        id = identifier or "default"
        result = self.delete_limiter.check(id)
        
        if result.allowed:
            success = self.cache.delete(key)
            return success, result
        else:
            raise RateLimitExceededError(
                f"Rate limit exceeded for delete operation. Retry after {result.retry_after:.2f} seconds"
            )


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""
    pass














