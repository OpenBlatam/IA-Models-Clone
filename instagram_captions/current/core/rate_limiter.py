"""
Rate Limiter for Instagram Captions API v10.0

Efficient rate limiting functionality.
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque

class RateLimiter:
    """Simple and efficient rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Get requests for this identifier
        requests = self.requests[identifier]
        
        # Remove old requests outside window
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < self.max_requests:
            requests.append(current_time)
            return True
        
        return False
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests allowed."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        requests = self.requests[identifier]
        
        # Remove old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        return max(0, self.max_requests - len(requests))
    
    def reset(self, identifier: Optional[str] = None):
        """Reset rate limiter for specific identifier or all."""
        if identifier:
            if identifier in self.requests:
                self.requests[identifier].clear()
        else:
            self.requests.clear()






