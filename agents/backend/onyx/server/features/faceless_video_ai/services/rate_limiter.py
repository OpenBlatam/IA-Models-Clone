"""
Rate Limiter Service
Implements rate limiting for API endpoints
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.limits: Dict[str, tuple] = {}  # (max_requests, window_seconds)
    
    def set_limit(self, key: str, max_requests: int, window_seconds: int):
        """Set rate limit for a key"""
        self.limits[key] = (max_requests, window_seconds)
        logger.info(f"Rate limit set for {key}: {max_requests} requests per {window_seconds}s")
    
    def is_allowed(self, key: str) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Check if request is allowed
        
        Returns:
            (is_allowed, info_dict) where info_dict contains:
            - remaining: remaining requests
            - reset_at: timestamp when limit resets
        """
        if key not in self.limits:
            return True, None
        
        max_requests, window_seconds = self.limits[key]
        now = time.time()
        
        # Clean old requests outside the window
        request_times = self.requests[key]
        while request_times and request_times[0] < now - window_seconds:
            request_times.popleft()
        
        # Check if limit exceeded
        if len(request_times) >= max_requests:
            reset_at = int(request_times[0] + window_seconds) if request_times else int(now + window_seconds)
            return False, {
                "remaining": 0,
                "reset_at": reset_at,
                "limit": max_requests,
                "window": window_seconds,
            }
        
        # Record this request
        request_times.append(now)
        
        reset_at = int(request_times[0] + window_seconds) if request_times else int(now + window_seconds)
        return True, {
            "remaining": max_requests - len(request_times),
            "reset_at": reset_at,
            "limit": max_requests,
            "window": window_seconds,
        }
    
    def reset(self, key: Optional[str] = None):
        """Reset rate limit for a key or all keys"""
        if key:
            self.requests[key].clear()
        else:
            self.requests.clear()


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance (singleton)"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        # Set default limits
        _rate_limiter.set_limit("default", max_requests=100, window_seconds=3600)  # 100 per hour
        _rate_limiter.set_limit("generate", max_requests=10, window_seconds=3600)  # 10 videos per hour
        _rate_limiter.set_limit("batch", max_requests=5, window_seconds=3600)  # 5 batches per hour
    return _rate_limiter

