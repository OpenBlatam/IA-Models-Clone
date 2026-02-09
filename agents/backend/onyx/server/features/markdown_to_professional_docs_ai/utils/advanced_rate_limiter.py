"""Advanced rate limiting per user"""
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class UserRateLimiter:
    """Rate limiter per user"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Requests per minute
            requests_per_hour: Requests per hour
            requests_per_day: Requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        
        # User request tracking
        self.user_requests: Dict[str, list] = defaultdict(list)
        self.user_limits: Dict[str, Dict[str, int]] = {}
    
    def set_user_limits(
        self,
        user_id: str,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None
    ):
        """
        Set custom limits for user
        
        Args:
            user_id: User ID
            requests_per_minute: Optional custom limit
            requests_per_hour: Optional custom limit
            requests_per_day: Optional custom limit
        """
        self.user_limits[user_id] = {
            "per_minute": requests_per_minute or self.requests_per_minute,
            "per_hour": requests_per_hour or self.requests_per_hour,
            "per_day": requests_per_day or self.requests_per_day
        }
    
    def is_allowed(
        self,
        user_id: str,
        identifier: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed
        
        Args:
            user_id: User ID
            identifier: Optional identifier (IP, API key, etc.)
            
        Returns:
            (is_allowed, reason_if_not)
        """
        # Get user limits
        limits = self.user_limits.get(user_id, {
            "per_minute": self.requests_per_minute,
            "per_hour": self.requests_per_hour,
            "per_day": self.requests_per_day
        })
        
        # Clean old requests
        now = datetime.now()
        user_requests = self.user_requests[user_id]
        user_requests[:] = [
            req_time for req_time in user_requests
            if (now - req_time).total_seconds() < 86400  # Keep last 24 hours
        ]
        
        # Check limits
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        requests_last_minute = sum(1 for req_time in user_requests if req_time > minute_ago)
        requests_last_hour = sum(1 for req_time in user_requests if req_time > hour_ago)
        requests_last_day = sum(1 for req_time in user_requests if req_time > day_ago)
        
        if requests_last_minute >= limits["per_minute"]:
            return False, "Rate limit exceeded: too many requests per minute"
        
        if requests_last_hour >= limits["per_hour"]:
            return False, "Rate limit exceeded: too many requests per hour"
        
        if requests_last_day >= limits["per_day"]:
            return False, "Rate limit exceeded: too many requests per day"
        
        # Record request
        user_requests.append(now)
        
        return True, None
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get user rate limit statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics
        """
        now = datetime.now()
        user_requests = self.user_requests[user_id]
        
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        limits = self.user_limits.get(user_id, {
            "per_minute": self.requests_per_minute,
            "per_hour": self.requests_per_hour,
            "per_day": self.requests_per_day
        })
        
        return {
            "user_id": user_id,
            "requests_last_minute": sum(1 for req_time in user_requests if req_time > minute_ago),
            "requests_last_hour": sum(1 for req_time in user_requests if req_time > hour_ago),
            "requests_last_day": sum(1 for req_time in user_requests if req_time > day_ago),
            "limits": limits
        }
    
    def reset_user(self, user_id: str):
        """Reset user request history"""
        if user_id in self.user_requests:
            del self.user_requests[user_id]


# Global rate limiter
_user_rate_limiter: Optional[UserRateLimiter] = None


def get_user_rate_limiter() -> UserRateLimiter:
    """Get global user rate limiter"""
    global _user_rate_limiter
    if _user_rate_limiter is None:
        from config import settings
        _user_rate_limiter = UserRateLimiter(
            requests_per_minute=getattr(settings, 'rate_limit_per_minute', 60),
            requests_per_hour=getattr(settings, 'rate_limit_per_hour', 1000),
            requests_per_day=getattr(settings, 'rate_limit_per_day', 10000)
        )
    return _user_rate_limiter

