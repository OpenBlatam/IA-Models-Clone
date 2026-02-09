"""
Rate limiting utilities for professional documents module.

Simple token bucket rate limiter for API endpoints.
"""

import time
from typing import Optional
from collections import defaultdict


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
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
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def available_tokens(self) -> float:
        """Get number of available tokens."""
        self._refill()
        return self.tokens


class RateLimiter:
    """Rate limiter with per-key token buckets."""
    
    def __init__(self, capacity: int = 100, refill_rate: float = 10.0):
        """
        Initialize rate limiter.
        
        Args:
            capacity: Maximum tokens per bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(capacity, refill_rate)
        )
    
    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed for the given key.
        
        Args:
            key: Rate limit key (e.g., user ID, IP address)
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        bucket = self.buckets[key]
        return bucket.consume(tokens)
    
    def get_remaining(self, key: str) -> float:
        """
        Get remaining tokens for a key.
        
        Args:
            key: Rate limit key
            
        Returns:
            Number of remaining tokens
        """
        return self.buckets[key].available_tokens()
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset rate limiter for a key or all keys.
        
        Args:
            key: Key to reset, or None to reset all
        """
        if key:
            self.buckets.pop(key, None)
        else:
            self.buckets.clear()


# Global rate limiter instance
default_rate_limiter = RateLimiter(capacity=100, refill_rate=10.0)






