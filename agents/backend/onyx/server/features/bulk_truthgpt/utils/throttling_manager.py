"""
Throttling Manager
==================

Advanced request throttling with multiple strategies and user-based limits.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class ThrottlingManager:
    """Advanced throttling manager."""
    
    def __init__(self):
        self.user_limits: Dict[str, Dict[str, Any]] = {}
        self.global_limits: Dict[str, Any] = {}
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.window_starts: Dict[str, Dict[str, datetime]] = defaultdict(lambda: defaultdict(lambda: datetime.now()))
        self.blocked_users: Dict[str, datetime] = {}
    
    def set_user_limit(
        self,
        user_id: str,
        requests_per_minute: int,
        requests_per_hour: int = None,
        requests_per_day: int = None
    ):
        """Set throttling limits for a user."""
        self.user_limits[user_id] = {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour,
            "per_day": requests_per_day
        }
        logger.info(f"Throttling limits set for user: {user_id}")
    
    def set_global_limit(
        self,
        requests_per_minute: int,
        requests_per_hour: int = None,
        requests_per_day: int = None
    ):
        """Set global throttling limits."""
        self.global_limits = {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour,
            "per_day": requests_per_day
        }
        logger.info("Global throttling limits set")
    
    def block_user(self, user_id: str, duration_minutes: int = 60):
        """Block a user for a duration."""
        self.blocked_users[user_id] = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"User blocked: {user_id} for {duration_minutes} minutes")
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked."""
        if user_id not in self.blocked_users:
            return False
        
        block_until = self.blocked_users[user_id]
        if datetime.now() > block_until:
            del self.blocked_users[user_id]
            return False
        
        return True
    
    async def check_throttle(
        self,
        user_id: Optional[str] = None,
        endpoint: str = "default"
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request should be throttled.
        
        Returns:
            (allowed, reason_if_blocked)
        """
        # Check if user is blocked
        if user_id and self.is_user_blocked(user_id):
            return False, "User is temporarily blocked"
        
        now = datetime.now()
        key = f"{user_id or 'global'}:{endpoint}"
        
        # Get limits
        limits = self.user_limits.get(user_id, {}) if user_id else self.global_limits
        
        # Check per-minute limit
        if "per_minute" in limits:
            window_key = f"{key}:minute"
            window_start = self.window_starts[window_key]
            
            # Reset window if needed
            if (now - window_start).total_seconds() >= 60:
                self.request_counts[window_key] = 0
                self.window_starts[window_key] = now
            
            if self.request_counts[window_key] >= limits["per_minute"]:
                return False, "Rate limit exceeded: per minute"
            
            self.request_counts[window_key] += 1
        
        # Check per-hour limit
        if "per_hour" in limits:
            window_key = f"{key}:hour"
            window_start = self.window_starts[window_key]
            
            # Reset window if needed
            if (now - window_start).total_seconds() >= 3600:
                self.request_counts[window_key] = 0
                self.window_starts[window_key] = now
            
            if self.request_counts[window_key] >= limits["per_hour"]:
                return False, "Rate limit exceeded: per hour"
            
            self.request_counts[window_key] += 1
        
        # Check per-day limit
        if "per_day" in limits:
            window_key = f"{key}:day"
            window_start = self.window_starts[window_key]
            
            # Reset window if needed
            if (now - window_start).total_seconds() >= 86400:
                self.request_counts[window_key] = 0
                self.window_starts[window_key] = now
            
            if self.request_counts[window_key] >= limits["per_day"]:
                return False, "Rate limit exceeded: per day"
            
            self.request_counts[window_key] += 1
        
        return True, None
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get throttling statistics for a user."""
        stats = {
            "is_blocked": self.is_user_blocked(user_id),
            "limits": self.user_limits.get(user_id, {}),
            "request_counts": {}
        }
        
        for key, count in self.request_counts.items():
            if user_id in key:
                stats["request_counts"][key] = count
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttling statistics."""
        return {
            "total_users": len(self.user_limits),
            "blocked_users": len(self.blocked_users),
            "global_limits": self.global_limits,
            "total_tracked_keys": len(self.request_counts)
        }

# Global instance
throttling_manager = ThrottlingManager()

















