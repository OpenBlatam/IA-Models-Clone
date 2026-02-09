"""
Rate limiting system
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = defaultdict(lambda: deque())
    
    def _cleanup_old_requests(self, client_id: str, current_time: float) -> None:
        """Remove old requests outside the time window"""
        client_requests = self._requests[client_id]
        cutoff_time = current_time - self.window_seconds
        
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed for client
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        
        client_requests = self._requests[client_id]
        
        if len(client_requests) >= self.max_requests:
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset_time": int(client_requests[0] + self.window_seconds),
                "retry_after": int(client_requests[0] + self.window_seconds - current_time)
            }
        
        # Add current request
        client_requests.append(current_time)
        
        return True, {
            "limit": self.max_requests,
            "remaining": self.max_requests - len(client_requests),
            "reset_time": int(current_time + self.window_seconds),
            "retry_after": 0
        }
    
    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """Get rate limit stats for a client"""
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        
        client_requests = self._requests[client_id]
        
        return {
            "requests": len(client_requests),
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - len(client_requests)),
            "reset_time": int(current_time + self.window_seconds)
        }


class IPRateLimiter:
    """Rate limiter based on IP address"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.rate_limiter = RateLimiter(max_requests, window_seconds)
    
    def is_allowed(self, ip_address: str) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed for IP address"""
        return self.rate_limiter.is_allowed(ip_address)
    
    def get_ip_stats(self, ip_address: str) -> Dict[str, int]:
        """Get rate limit stats for IP address"""
        return self.rate_limiter.get_client_stats(ip_address)


class EndpointRateLimiter:
    """Rate limiter for specific endpoints"""
    
    def __init__(self):
        self._limiters: Dict[str, IPRateLimiter] = {}
        self._default_limits = {
            "/analyze": (50, 60),      # 50 requests per minute
            "/similarity": (50, 60),   # 50 requests per minute
            "/quality": (50, 60),      # 50 requests per minute
            "/health": (100, 60),      # 100 requests per minute
            "/stats": (20, 60),        # 20 requests per minute
        }
    
    def _get_limiter(self, endpoint: str) -> IPRateLimiter:
        """Get or create rate limiter for endpoint"""
        if endpoint not in self._limiters:
            max_requests, window_seconds = self._default_limits.get(endpoint, (100, 60))
            self._limiters[endpoint] = IPRateLimiter(max_requests, window_seconds)
        
        return self._limiters[endpoint]
    
    def is_allowed(self, ip_address: str, endpoint: str) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed for IP and endpoint"""
        limiter = self._get_limiter(endpoint)
        return limiter.is_allowed(ip_address)
    
    def get_endpoint_stats(self, ip_address: str, endpoint: str) -> Dict[str, int]:
        """Get rate limit stats for IP and endpoint"""
        limiter = self._get_limiter(endpoint)
        return limiter.get_ip_stats(ip_address)


# Global rate limiter
endpoint_rate_limiter = EndpointRateLimiter()


def check_rate_limit(ip_address: str, endpoint: str) -> Tuple[bool, Dict[str, int]]:
    """Check rate limit for IP and endpoint"""
    return endpoint_rate_limiter.is_allowed(ip_address, endpoint)


def get_rate_limit_stats(ip_address: str, endpoint: str) -> Dict[str, int]:
    """Get rate limit stats for IP and endpoint"""
    return endpoint_rate_limiter.get_endpoint_stats(ip_address, endpoint)


