"""Rate Limiting for Markdown to Professional Documents AI"""
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str = "default") -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests
            self._requests[identifier] = [
                req_time for req_time in self._requests[identifier]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self._requests[identifier]) >= self.max_requests:
                return False, 0
            
            # Add current request
            self._requests[identifier].append(now)
            
            remaining = self.max_requests - len(self._requests[identifier])
            return True, remaining
    
    def get_remaining(self, identifier: str = "default") -> int:
        """Get remaining requests for identifier"""
        with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests
            self._requests[identifier] = [
                req_time for req_time in self._requests[identifier]
                if req_time > window_start
            ]
            
            return max(0, self.max_requests - len(self._requests[identifier]))
    
    def reset(self, identifier: Optional[str] = None) -> None:
        """Reset rate limit for identifier or all"""
        with self._lock:
            if identifier:
                self._requests[identifier].clear()
            else:
                self._requests.clear()


# Global rate limiter instance
_rate_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter(max_requests: int = 100, window_seconds: int = 60) -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter(max_requests, window_seconds)
    return _rate_limiter_instance

