"""
Rate Limiter for Autonomous SAM3 Agent
=======================================

Rate limiting to prevent API overload and manage resource usage.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 100  # Max requests per window
    window_seconds: int = 60  # Time window in seconds
    max_concurrent: int = 10  # Max concurrent requests


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self.request_times: deque = deque()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request.
        
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.request_times and self.request_times[0] < now - self.config.window_seconds:
                self.request_times.popleft()
            
            # Check if we've exceeded the limit
            if len(self.request_times) >= self.config.max_requests:
                oldest_request = self.request_times[0]
                wait_time = (oldest_request + self.config.window_seconds) - now
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Retry after waiting
                    return await self.acquire()
            
            # Add current request
            self.request_times.append(now)
        
        # Acquire semaphore for concurrent limit
        await self.semaphore.acquire()
    
    async def release(self) -> None:
        """Release semaphore after request completes."""
        self.semaphore.release()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        recent_requests = [
            t for t in self.request_times
            if t >= now - self.config.window_seconds
        ]
        
        return {
            "requests_in_window": len(recent_requests),
            "max_requests": self.config.max_requests,
            "window_seconds": self.config.window_seconds,
            "available_slots": self.config.max_requests - len(recent_requests),
            "concurrent_requests": self.config.max_concurrent - self.semaphore._value,
            "max_concurrent": self.config.max_concurrent,
        }


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""
    pass
