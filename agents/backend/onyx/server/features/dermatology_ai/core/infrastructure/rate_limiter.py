"""
Rate Limiting Implementation
Prevents abuse and ensures fair resource usage
"""

import time
from typing import Dict, Optional
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter using token bucket algorithm
    For production, consider using Redis-based rate limiter
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to requests_per_second)
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.tokens: Dict[str, float] = defaultdict(lambda: float(self.burst_size))
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed
        
        Args:
            key: Unique identifier for rate limiting (e.g., user_id, ip_address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self._lock:
            now = time.time()
            last = self.last_update[key]
            
            # Add tokens based on time passed
            time_passed = now - last
            tokens_to_add = time_passed * self.requests_per_second
            self.tokens[key] = min(
                self.burst_size,
                self.tokens[key] + tokens_to_add
            )
            self.last_update[key] = now
            
            # Check if we have tokens
            if self.tokens[key] >= 1.0:
                self.tokens[key] -= 1.0
                return True
            
            return False
    
    async def get_remaining(self, key: str) -> int:
        """Get remaining requests for key"""
        async with self._lock:
            now = time.time()
            last = self.last_update.get(key, now)
            
            time_passed = now - last
            tokens_to_add = time_passed * self.requests_per_second
            current_tokens = min(
                self.burst_size,
                self.tokens.get(key, self.burst_size) + tokens_to_add
            )
            
            return int(current_tokens)
    
    def reset(self, key: Optional[str] = None):
        """Reset rate limiter for key or all keys"""
        if key:
            self.tokens.pop(key, None)
            self.last_update.pop(key, None)
        else:
            self.tokens.clear()
            self.last_update.clear()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass















