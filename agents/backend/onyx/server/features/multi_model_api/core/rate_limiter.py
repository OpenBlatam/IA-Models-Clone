"""
Rate Limiter for Multi-Model API
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limit information"""
    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: Optional[int] = None


class RateLimiter:
    """Rate limiter with sliding window algorithm"""
    
    def __init__(
        self,
        default_limit: int = 100,
        default_window: int = 60,
        burst_limit: int = 10
    ):
        self.default_limit = default_limit
        self.default_window = default_window
        self.burst_limit = burst_limit
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.limits: Dict[str, Tuple[int, int]] = {}
        self._lock = asyncio.Lock()
    
    def set_limit(self, key: str, max_requests: int, window_seconds: int):
        """Set custom rate limit for a key"""
        self.limits[key] = (max_requests, window_seconds)
        logger.info(f"Rate limit set for {key}: {max_requests} requests per {window_seconds}s")
    
    async def is_allowed(
        self,
        identifier: str,
        endpoint: str = "default",
        custom_limit: Optional[int] = None,
        custom_window: Optional[int] = None
    ) -> RateLimitInfo:
        """
        Check if request is allowed - optimized for speed
        
        Args:
            identifier: Client identifier (IP, API key, etc.)
            endpoint: Endpoint name
            custom_limit: Custom limit override
            custom_window: Custom window override
            
        Returns:
            RateLimitInfo with rate limit status
        """
        key = f"{identifier}:{endpoint}"
        now = time.time()
        
        max_requests, window_seconds = self.limits.get(
            endpoint,
            (custom_limit or self.default_limit, custom_window or self.default_window)
        )
        
        async with self._lock:
            request_times = self.requests[key]
            burst_times = self.burst_requests[key]
            
            window_start = now - window_seconds
            burst_window_start = now - 1
            
            if request_times:
                while request_times and request_times[0] < window_start:
                    request_times.popleft()
            
            if burst_times:
                while burst_times and burst_times[0] < burst_window_start:
                    burst_times.popleft()
            
            burst_count = len(burst_times)
            if burst_count >= self.burst_limit:
                reset_at = burst_times[0] + 1 if burst_times else now + 1
                retry_after = int(reset_at - now) + 1
                return RateLimitInfo(
                    allowed=False,
                    remaining=0,
                    limit=max_requests,
                    reset_at=reset_at,
                    retry_after=retry_after
                )
            
            request_count = len(request_times)
            if request_count >= max_requests:
                reset_at = request_times[0] + window_seconds if request_times else now + window_seconds
                retry_after = int(reset_at - now) + 1
                return RateLimitInfo(
                    allowed=False,
                    remaining=0,
                    limit=max_requests,
                    reset_at=reset_at,
                    retry_after=retry_after
                )
            
            request_times.append(now)
            burst_times.append(now)
            
            remaining = max_requests - request_count - 1
            reset_at = request_times[0] + window_seconds if request_times else now + window_seconds
            
            return RateLimitInfo(
                allowed=True,
                remaining=remaining,
                limit=max_requests,
                reset_at=reset_at
            )
    
    async def get_rate_limit_info(
        self,
        identifier: str,
        endpoint: str = "default"
    ) -> Dict[str, Any]:
        """Get current rate limit information without consuming a request"""
        key = f"{identifier}:{endpoint}"
        now = time.time()
        
        max_requests, window_seconds = self.limits.get(
            endpoint,
            (self.default_limit, self.default_window)
        )
        
        async with self._lock:
            request_times = self.requests[key]
            window_start = now - window_seconds
            
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            remaining = max(0, max_requests - len(request_times))
            reset_at = request_times[0] + window_seconds if request_times else now + window_seconds
            
            return {
                "limit": max_requests,
                "remaining": remaining,
                "reset_at": reset_at,
                "window_seconds": window_seconds
            }
    
    def reset_limit(self, identifier: str, endpoint: str = "default"):
        """Reset rate limit for a specific identifier"""
        key = f"{identifier}:{endpoint}"
        if key in self.requests:
            del self.requests[key]
        if key in self.burst_requests:
            del self.burst_requests[key]
        logger.info(f"Rate limit reset for {key}")


_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter

