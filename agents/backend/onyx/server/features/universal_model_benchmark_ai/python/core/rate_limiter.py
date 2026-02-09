"""
Rate Limiter Module - Rate limiting and throttling.

Provides:
- Token bucket algorithm
- Sliding window rate limiting
- Per-user rate limits
- Distributed rate limiting support
"""

import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window_seconds: int
    burst: Optional[int] = None  # Optional burst allowance


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def available(self) -> int:
        """Get available tokens."""
        with self.lock:
            self._refill()
            return int(self.tokens)


class SlidingWindowLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, requests: int, window_seconds: int):
        """
        Initialize sliding window limiter.
        
        Args:
            requests: Maximum requests
            window_seconds: Time window in seconds
        """
        self.requests = requests
        self.window_seconds = window_seconds
        self.requests_history: deque = deque()
        self.lock = Lock()
    
    def is_allowed(self) -> bool:
        """
        Check if request is allowed.
        
        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            cutoff = now - self.window_seconds
            while self.requests_history and self.requests_history[0] < cutoff:
                self.requests_history.popleft()
            
            # Check if under limit
            if len(self.requests_history) < self.requests:
                self.requests_history.append(now)
                return True
            
            return False
    
    def remaining(self) -> int:
        """Get remaining requests in window."""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            while self.requests_history and self.requests_history[0] < cutoff:
                self.requests_history.popleft()
            return max(0, self.requests - len(self.requests_history))


class RateLimiter:
    """Rate limiter manager."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.limiters: Dict[str, Dict[str, any]] = {}
        self.default_limit = RateLimit(requests=100, window_seconds=60)
        self.lock = Lock()
    
    def get_limiter(
        self,
        key: str,
        limit: Optional[RateLimit] = None,
        algorithm: str = "sliding_window",
    ) -> any:
        """
        Get or create rate limiter for key.
        
        Args:
            key: Limiter key (e.g., user_id, IP)
            limit: Rate limit configuration
            algorithm: Algorithm to use (token_bucket, sliding_window)
            
        Returns:
            Rate limiter instance
        """
        if limit is None:
            limit = self.default_limit
        
        with self.lock:
            if key not in self.limiters:
                if algorithm == "token_bucket":
                    # Calculate refill rate
                    refill_rate = limit.requests / limit.window_seconds
                    capacity = limit.burst or limit.requests
                    self.limiters[key] = {
                        "limiter": TokenBucket(capacity, refill_rate),
                        "limit": limit,
                    }
                else:  # sliding_window
                    self.limiters[key] = {
                        "limiter": SlidingWindowLimiter(
                            limit.requests,
                            limit.window_seconds
                        ),
                        "limit": limit,
                    }
            
            return self.limiters[key]["limiter"]
    
    def check_rate_limit(
        self,
        key: str,
        limit: Optional[RateLimit] = None,
        algorithm: str = "sliding_window",
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Limiter key
            limit: Rate limit configuration
            algorithm: Algorithm to use
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        limiter = self.get_limiter(key, limit, algorithm)
        
        if algorithm == "token_bucket":
            allowed = limiter.consume()
            remaining = limiter.available()
        else:  # sliding_window
            allowed = limiter.is_allowed()
            remaining = limiter.remaining()
        
        return allowed, remaining
    
    def reset_limiter(self, key: str) -> None:
        """Reset limiter for key."""
        with self.lock:
            if key in self.limiters:
                del self.limiters[key]
    
    def get_stats(self, key: str) -> Optional[Dict]:
        """Get limiter statistics."""
        with self.lock:
            if key in self.limiters:
                limiter = self.limiters[key]["limiter"]
                limit = self.limiters[key]["limit"]
                
                if isinstance(limiter, SlidingWindowLimiter):
                    remaining = limiter.remaining()
                else:
                    remaining = limiter.available()
                
                return {
                    "limit": {
                        "requests": limit.requests,
                        "window_seconds": limit.window_seconds,
                    },
                    "remaining": remaining,
                }
            return None

