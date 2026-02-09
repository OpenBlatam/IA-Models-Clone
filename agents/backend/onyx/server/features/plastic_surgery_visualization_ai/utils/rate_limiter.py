"""Rate limiting utilities."""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from utils.logger import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.utcnow()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def wait_for_token(self, tokens: int = 1, timeout: float = 60.0) -> bool:
        """
        Wait for tokens to become available.
        
        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        start = datetime.utcnow()
        
        while (datetime.utcnow() - start).total_seconds() < timeout:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)
        
        return False


class RateLimiter:
    """Rate limiter with multiple buckets."""
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
    
    def create_bucket(
        self,
        key: str,
        capacity: int,
        refill_rate: float
    ) -> TokenBucket:
        """
        Create a rate limit bucket.
        
        Args:
            key: Unique key for the bucket
            capacity: Maximum tokens
            refill_rate: Tokens per second
            
        Returns:
            TokenBucket instance
        """
        bucket = TokenBucket(capacity, refill_rate)
        self.buckets[key] = bucket
        return bucket
    
    async def is_allowed(
        self,
        key: str,
        tokens: int = 1
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Bucket key
            tokens: Number of tokens needed
            
        Returns:
            True if allowed, False otherwise
        """
        async with self._lock:
            bucket = self.buckets.get(key)
            if not bucket:
                return True  # No limit if bucket doesn't exist
            
            return await bucket.acquire(tokens)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

