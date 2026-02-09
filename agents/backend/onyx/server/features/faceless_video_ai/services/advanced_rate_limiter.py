"""
Advanced Rate Limiter
Per-user rate limiting with quotas
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class UserQuota:
    """User quota information"""
    
    def __init__(
        self,
        user_id: str,
        daily_limit: int = 100,
        hourly_limit: int = 20,
        monthly_limit: int = 1000
    ):
        self.user_id = user_id
        self.daily_limit = daily_limit
        self.hourly_limit = hourly_limit
        self.monthly_limit = monthly_limit
        self.daily_count = 0
        self.hourly_count = 0
        self.monthly_count = 0
        self.last_reset_daily = datetime.utcnow().date()
        self.last_reset_hourly = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        self.last_reset_monthly = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def reset_if_needed(self):
        """Reset counters if period has passed"""
        now = datetime.utcnow()
        
        # Reset daily
        if now.date() > self.last_reset_daily:
            self.daily_count = 0
            self.last_reset_daily = now.date()
        
        # Reset hourly
        if now >= self.last_reset_hourly + timedelta(hours=1):
            self.hourly_count = 0
            self.last_reset_hourly = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset monthly
        if now.month != self.last_reset_monthly.month or now.year != self.last_reset_monthly.year:
            self.monthly_count = 0
            self.last_reset_monthly = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def is_allowed(self) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Check if request is allowed"""
        self.reset_if_needed()
        
        if self.daily_count >= self.daily_limit:
            return False, {
                "limit": self.daily_limit,
                "remaining": 0,
                "reset_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
                "period": "daily"
            }
        
        if self.hourly_count >= self.hourly_limit:
            return False, {
                "limit": self.hourly_limit,
                "remaining": 0,
                "reset_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "period": "hourly"
            }
        
        if self.monthly_count >= self.monthly_limit:
            return False, {
                "limit": self.monthly_limit,
                "remaining": 0,
                "reset_at": (datetime.utcnow().replace(day=1) + timedelta(days=32)).replace(day=1).isoformat(),
                "period": "monthly"
            }
        
        return True, {
            "daily_remaining": self.daily_limit - self.daily_count,
            "hourly_remaining": self.hourly_limit - self.hourly_count,
            "monthly_remaining": self.monthly_limit - self.monthly_count,
        }
    
    def increment(self):
        """Increment counters"""
        self.reset_if_needed()
        self.daily_count += 1
        self.hourly_count += 1
        self.monthly_count += 1


class AdvancedRateLimiter:
    """Advanced rate limiter with per-user quotas"""
    
    def __init__(self):
        self.quotas: Dict[str, UserQuota] = {}
        self.default_limits = {
            "daily": 100,
            "hourly": 20,
            "monthly": 1000,
        }
    
    def get_or_create_quota(self, user_id: str) -> UserQuota:
        """Get or create user quota"""
        if user_id not in self.quotas:
            self.quotas[user_id] = UserQuota(
                user_id=user_id,
                daily_limit=self.default_limits["daily"],
                hourly_limit=self.default_limits["hourly"],
                monthly_limit=self.default_limits["monthly"]
            )
        return self.quotas[user_id]
    
    def set_user_limits(
        self,
        user_id: str,
        daily_limit: Optional[int] = None,
        hourly_limit: Optional[int] = None,
        monthly_limit: Optional[int] = None
    ):
        """Set custom limits for user"""
        quota = self.get_or_create_quota(user_id)
        
        if daily_limit is not None:
            quota.daily_limit = daily_limit
        if hourly_limit is not None:
            quota.hourly_limit = hourly_limit
        if monthly_limit is not None:
            quota.monthly_limit = monthly_limit
        
        logger.info(f"Updated limits for user {user_id}")
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Check if user can make request
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (allowed, quota_info)
        """
        quota = self.get_or_create_quota(user_id)
        return quota.is_allowed()
    
    def record_request(self, user_id: str):
        """Record that user made a request"""
        quota = self.get_or_create_quota(user_id)
        quota.increment()
    
    def get_quota_info(self, user_id: str) -> Dict[str, Any]:
        """Get quota information for user"""
        quota = self.get_or_create_quota(user_id)
        quota.reset_if_needed()
        
        return {
            "user_id": user_id,
            "limits": {
                "daily": quota.daily_limit,
                "hourly": quota.hourly_limit,
                "monthly": quota.monthly_limit,
            },
            "usage": {
                "daily": quota.daily_count,
                "hourly": quota.hourly_count,
                "monthly": quota.monthly_count,
            },
            "remaining": {
                "daily": quota.daily_limit - quota.daily_count,
                "hourly": quota.hourly_limit - quota.hourly_count,
                "monthly": quota.monthly_limit - quota.monthly_count,
            },
        }


_advanced_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_advanced_rate_limiter() -> AdvancedRateLimiter:
    """Get advanced rate limiter instance (singleton)"""
    global _advanced_rate_limiter
    if _advanced_rate_limiter is None:
        _advanced_rate_limiter = AdvancedRateLimiter()
    return _advanced_rate_limiter

