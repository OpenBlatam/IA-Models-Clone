"""
Rate Limiting for API endpoints
"""

import asyncio
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using token bucket algorithm"""
    
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
        Check if request is allowed
        
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
        """Get rate limit statistics"""
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




