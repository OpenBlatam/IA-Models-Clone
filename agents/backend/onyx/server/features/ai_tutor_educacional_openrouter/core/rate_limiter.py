"""
Rate limiter for API calls to prevent exceeding limits.
"""

import asyncio
import logging
from typing import Dict
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    """
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self.lock:
            now = datetime.now()
            
            # Remove old requests outside time window
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            # Check if we can make a request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = self.time_window - (now - oldest_request).total_seconds()
            
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                return await self.acquire()
            
            return False
    
    def get_stats(self) -> Dict:
        """Get current rate limiter statistics."""
        now = datetime.now()
        recent_requests = [
            req for req in self.requests
            if (now - req).total_seconds() <= self.time_window
        ]
        
        return {
            "current_requests": len(recent_requests),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "available_slots": self.max_requests - len(recent_requests)
        }






