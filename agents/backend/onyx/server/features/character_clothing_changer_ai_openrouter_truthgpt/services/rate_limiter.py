"""
Rate Limiter Service
====================

Service for rate limiting API requests.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RATE_LIMIT = 100  # requests per window
DEFAULT_WINDOW_SECONDS = 60  # 1 minute
MAX_RATE_LIMIT = 10000
MIN_WINDOW_SECONDS = 1


@dataclass
class RateLimitInfo:
    """Rate limit information for a client"""
    client_id: str
    requests: deque
    limit: int
    window_seconds: float
    reset_at: datetime
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        
        # Reset if window expired
        if now > self.reset_at:
            self.requests.clear()
            self.reset_at = now + timedelta(seconds=self.window_seconds)
            return True
        
        # Remove old requests outside window
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()
        
        # Check if under limit
        return len(self.requests) < self.limit
    
    def record_request(self) -> None:
        """Record a request"""
        now = datetime.now()
        self.requests.append(now)
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window"""
        now = datetime.now()
        
        # Reset if window expired
        if now > self.reset_at:
            return self.limit
        
        # Remove old requests
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()
        
        return max(0, self.limit - len(self.requests))
    
    def get_reset_in(self) -> float:
        """Get seconds until rate limit resets"""
        now = datetime.now()
        if now > self.reset_at:
            return 0.0
        return (self.reset_at - now).total_seconds()


class RateLimiter:
    """
    Rate limiter for API requests.
    
    Features:
    - Per-client rate limiting
    - Sliding window algorithm
    - Configurable limits and windows
    - Request tracking
    """
    
    def __init__(
        self,
        default_limit: int = DEFAULT_RATE_LIMIT,
        default_window: float = DEFAULT_WINDOW_SECONDS
    ):
        """
        Initialize rate limiter.
        
        Args:
            default_limit: Default requests per window (default: 100)
            default_window: Default window size in seconds (default: 60)
        """
        self.default_limit = min(default_limit, MAX_RATE_LIMIT)
        self.default_window = max(default_window, MIN_WINDOW_SECONDS)
        self.clients: Dict[str, RateLimitInfo] = {}
    
    def _get_client_id(self, identifier: Optional[str] = None) -> str:
        """
        Get or generate client identifier.
        
        Args:
            identifier: Optional client identifier
            
        Returns:
            Client ID string
        """
        return identifier or "default"
    
    def _get_or_create_client(
        self,
        client_id: str,
        limit: Optional[int] = None,
        window: Optional[float] = None
    ) -> RateLimitInfo:
        """
        Get or create rate limit info for client.
        
        Args:
            client_id: Client identifier
            limit: Optional custom limit
            window: Optional custom window
            
        Returns:
            RateLimitInfo object
        """
        if client_id not in self.clients:
            limit = limit or self.default_limit
            window = window or self.default_window
            now = datetime.now()
            
            self.clients[client_id] = RateLimitInfo(
                client_id=client_id,
                requests=deque(),
                limit=limit,
                window_seconds=window,
                reset_at=now + timedelta(seconds=window)
            )
        
        return self.clients[client_id]
    
    def is_allowed(
        self,
        client_id: Optional[str] = None,
        limit: Optional[int] = None,
        window: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Args:
            client_id: Optional client identifier
            limit: Optional custom limit
            window: Optional custom window
            
        Returns:
            Tuple of (is_allowed, info_dict):
            - is_allowed: bool - Whether request is allowed
            - info: dict - Rate limit information
        """
        client_id = self._get_client_id(client_id)
        client_info = self._get_or_create_client(client_id, limit, window)
        
        is_allowed = client_info.is_allowed()
        
        info = {
            "allowed": is_allowed,
            "client_id": client_id,
            "limit": client_info.limit,
            "remaining": client_info.get_remaining(),
            "reset_in": client_info.get_reset_in(),
            "window_seconds": client_info.window_seconds
        }
        
        if is_allowed:
            client_info.record_request()
        
        return is_allowed, info
    
    def get_info(
        self,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get rate limit information for client.
        
        Args:
            client_id: Optional client identifier
            
        Returns:
            Dictionary with rate limit information
        """
        client_id = self._get_client_id(client_id)
        
        if client_id not in self.clients:
            return {
                "client_id": client_id,
                "limit": self.default_limit,
                "remaining": self.default_limit,
                "reset_in": 0.0,
                "window_seconds": self.default_window
            }
        
        client_info = self.clients[client_id]
        
        return {
            "client_id": client_id,
            "limit": client_info.limit,
            "remaining": client_info.get_remaining(),
            "reset_in": client_info.get_reset_in(),
            "window_seconds": client_info.window_seconds,
            "requests_in_window": len(client_info.requests)
        }
    
    def reset_client(
        self,
        client_id: Optional[str] = None
    ) -> bool:
        """
        Reset rate limit for a client.
        
        Args:
            client_id: Optional client identifier
            
        Returns:
            True if client was reset, False if not found
        """
        client_id = self._get_client_id(client_id)
        
        if client_id in self.clients:
            del self.clients[client_id]
            logger.debug(f"Rate limit reset for client: {client_id}")
            return True
        
        return False
    
    def clear_all(self) -> int:
        """
        Clear all rate limit data.
        
        Returns:
            Number of clients cleared
        """
        count = len(self.clients)
        self.clients.clear()
        logger.info(f"Rate limiter cleared: {count} clients")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics:
            - total_clients: int
            - clients: List[Dict] - Client information
        """
        clients_info = []
        for client_id, client_info in self.clients.items():
            clients_info.append({
                "client_id": client_id,
                "limit": client_info.limit,
                "remaining": client_info.get_remaining(),
                "reset_in": client_info.get_reset_in(),
                "window_seconds": client_info.window_seconds,
                "requests_in_window": len(client_info.requests)
            })
        
        return {
            "total_clients": len(self.clients),
            "default_limit": self.default_limit,
            "default_window": self.default_window,
            "clients": clients_info
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

