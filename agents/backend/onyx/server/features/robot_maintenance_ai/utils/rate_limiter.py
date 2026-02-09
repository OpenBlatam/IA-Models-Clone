"""
Rate limiting utilities for API endpoints.
"""

import time
from collections import defaultdict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter using token bucket algorithm.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        logger.info(f"RateLimiter initialized: max_requests={max_requests}, window={window_seconds}s")
    
    def is_allowed(self, identifier: str = "default") -> tuple[bool, Optional[str]]:
        """
        Check if a request is allowed.
        
        Args:
            identifier: Unique identifier for the client (IP, user ID, etc.)
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        client_requests = self.requests[identifier]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < self.window_seconds]
        
        if len(client_requests) >= self.max_requests:
            oldest_request = min(client_requests) if client_requests else now
            retry_after = int(self.window_seconds - (now - oldest_request)) + 1
            return False, f"Rate limit exceeded. Try again in {retry_after} seconds."
        
        # Add current request
        client_requests.append(now)
        return True, None
    
    def reset(self, identifier: Optional[str] = None):
        """
        Reset rate limit for a specific identifier or all.
        
        Args:
            identifier: Client identifier to reset, or None to reset all
        """
        if identifier:
            if identifier in self.requests:
                del self.requests[identifier]
                logger.info(f"Rate limit reset for identifier: {identifier}")
        else:
            self.requests.clear()
            logger.info("All rate limits reset")
    
    def get_remaining(self, identifier: str = "default") -> int:
        """
        Get remaining requests for an identifier.
        
        Args:
            identifier: Client identifier
        
        Returns:
            Number of remaining requests
        """
        now = time.time()
        client_requests = self.requests[identifier]
        
        # Remove old requests
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < self.window_seconds]
        
        return max(0, self.max_requests - len(client_requests))
    
    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics
        """
        now = time.time()
        active_clients = 0
        total_requests = 0
        
        for identifier, requests in self.requests.items():
            # Clean old requests
            valid_requests = [r for r in requests if now - r < self.window_seconds]
            self.requests[identifier] = valid_requests
            
            if valid_requests:
                active_clients += 1
                total_requests += len(valid_requests)
        
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_clients": active_clients,
            "total_requests": total_requests
        }






