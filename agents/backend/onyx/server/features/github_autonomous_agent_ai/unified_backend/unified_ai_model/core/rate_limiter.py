"""
Rate Limiter Module
Simple rate limiting using sliding window algorithm.
Ported from autonomous_long_term_agent/core/rate_limiter.py
"""

import asyncio
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using sliding window algorithm."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str = "default") -> Tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Clean old requests
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > cutoff
            ]
            
            # Check if limit exceeded
            if len(self._requests[key]) >= self.max_requests:
                remaining = 0
                return False, remaining
            
            # Add current request
            self._requests[key].append(now)
            remaining = self.max_requests - len(self._requests[key])
            
            return True, remaining
    
    async def get_stats(self, key: str = "default") -> Dict[str, any]:
        """Get rate limit statistics."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            requests = [
                req_time for req_time in self._requests[key]
                if req_time > cutoff
            ]
            
            return {
                "requests_in_window": len(requests),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "remaining": max(0, self.max_requests - len(requests))
            }
    
    async def reset(self, key: str = "default") -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            self._requests[key] = []
    
    async def wait_if_needed(self, key: str = "default") -> bool:
        """
        Wait if rate limited, then proceed.
        
        Returns:
            True if had to wait, False otherwise
        """
        allowed, remaining = await self.is_allowed(key)
        
        if allowed:
            return False
        
        # Calculate wait time (oldest request + window - now)
        async with self._lock:
            if self._requests[key]:
                oldest = min(self._requests[key])
                wait_time = oldest + self.window_seconds - time.time()
                if wait_time > 0:
                    logger.info(f"Rate limited, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
        
        return True


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        key: str = None,
        remaining: int = 0,
        retry_after: int = None
    ):
        message = "Rate limit exceeded"
        if key:
            message += f" for '{key}'"
        if remaining >= 0:
            message += f". Remaining: {remaining}"
        if retry_after:
            message += f". Retry after {retry_after}s"
        
        super().__init__(message)
        self.key = key
        self.remaining = remaining
        self.retry_after = retry_after
