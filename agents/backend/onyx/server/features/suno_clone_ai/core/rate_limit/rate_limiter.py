"""
Rate Limiter

Utilities for rate limiting and throttling.
"""

import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with sliding window."""
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: float = 60.0
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(
        self,
        identifier: str = "default"
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            identifier: Request identifier (e.g., user ID, IP)
            
        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.time_window
            ]
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add request
            self.requests[identifier].append(now)
            return True
    
    def get_remaining(
        self,
        identifier: str = "default"
    ) -> int:
        """
        Get remaining requests.
        
        Args:
            identifier: Request identifier
            
        Returns:
            Number of remaining requests
        """
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.time_window
            ]
            
            return max(0, self.max_requests - len(self.requests[identifier]))
    
    def reset(self, identifier: Optional[str] = None) -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Request identifier (None = all)
        """
        with self.lock:
            if identifier is None:
                self.requests.clear()
            else:
                self.requests[identifier].clear()


def rate_limit(
    max_requests: int = 100,
    time_window: float = 60.0
):
    """
    Rate limit decorator.
    
    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds
        
    Returns:
        Decorator function
    """
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            identifier = kwargs.get('identifier', 'default')
            
            if not limiter.is_allowed(identifier):
                raise Exception(f"Rate limit exceeded for {identifier}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def throttle(
    max_requests: int = 100,
    time_window: float = 60.0
):
    """
    Throttle decorator (waits instead of raising error).
    
    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds
        
    Returns:
        Decorator function
    """
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            identifier = kwargs.get('identifier', 'default')
            
            while not limiter.is_allowed(identifier):
                time.sleep(0.1)  # Wait before retry
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator



