"""
Rate Limiter for Flux2 Clothing Changer
========================================

Rate limiting for API and processing requests.
"""

import time
import threading
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    max_requests: int
    window_seconds: float


class RateLimiter:
    """Rate limiter with sliding window."""
    
    def __init__(
        self,
        default_limit: RateLimit = RateLimit(max_requests=10, window_seconds=60.0),
    ):
        """
        Initialize rate limiter.
        
        Args:
            default_limit: Default rate limit
        """
        self.default_limit = default_limit
        self.limits: Dict[str, RateLimit] = {}
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def set_limit(self, key: str, limit: RateLimit) -> None:
        """
        Set rate limit for a key.
        
        Args:
            key: Identifier (e.g., user_id, ip_address)
            limit: Rate limit configuration
        """
        with self.lock:
            self.limits[key] = limit
    
    def check_limit(
        self,
        key: str,
        increment: bool = True,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Identifier
            increment: Whether to increment request count
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        with self.lock:
            limit = self.limits.get(key, self.default_limit)
            now = time.time()
            
            # Clean old requests
            requests = self.requests[key]
            while requests and requests[0] < now - limit.window_seconds:
                requests.popleft()
            
            # Check limit
            if len(requests) >= limit.max_requests:
                return False, {
                    "allowed": False,
                    "limit": limit.max_requests,
                    "window": limit.window_seconds,
                    "remaining": 0,
                    "reset_at": requests[0] + limit.window_seconds if requests else now,
                }
            
            # Increment if allowed
            if increment:
                requests.append(now)
            
            return True, {
                "allowed": True,
                "limit": limit.max_requests,
                "window": limit.window_seconds,
                "remaining": limit.max_requests - len(requests),
                "reset_at": requests[0] + limit.window_seconds if requests else now,
            }
    
    def get_status(self, key: str) -> Dict[str, Any]:
        """
        Get current rate limit status.
        
        Args:
            key: Identifier
            
        Returns:
            Status dictionary
        """
        with self.lock:
            limit = self.limits.get(key, self.default_limit)
            now = time.time()
            
            requests = self.requests[key]
            while requests and requests[0] < now - limit.window_seconds:
                requests.popleft()
            
            return {
                "limit": limit.max_requests,
                "window": limit.window_seconds,
                "used": len(requests),
                "remaining": limit.max_requests - len(requests),
                "reset_at": requests[0] + limit.window_seconds if requests else now,
            }
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset rate limit for a key or all keys.
        
        Args:
            key: Identifier (None for all)
        """
        with self.lock:
            if key:
                self.requests[key].clear()
            else:
                self.requests.clear()

