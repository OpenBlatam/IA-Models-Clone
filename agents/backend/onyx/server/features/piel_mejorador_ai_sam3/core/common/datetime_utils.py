"""
DateTime Utilities for Piel Mejorador AI SAM3
=============================================

Common datetime operations and utilities.
"""

import logging
from typing import Optional, Union
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class DateTimeUtils:
    """Unified datetime utilities."""
    
    @staticmethod
    def now() -> datetime:
        """Get current datetime with timezone."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def now_iso() -> str:
        """Get current datetime as ISO format string."""
        return DateTimeUtils.now().isoformat()
    
    @staticmethod
    def parse_iso(iso_string: str) -> datetime:
        """
        Parse ISO format datetime string.
        
        Args:
            iso_string: ISO format datetime string
            
        Returns:
            Datetime object
        """
        try:
            return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        except ValueError:
            # Try without timezone
            return datetime.fromisoformat(iso_string)
    
    @staticmethod
    def to_iso(dt: datetime) -> str:
        """
        Convert datetime to ISO format string.
        
        Args:
            dt: Datetime object
            
        Returns:
            ISO format string
        """
        return dt.isoformat()
    
    @staticmethod
    def elapsed(start: datetime, end: Optional[datetime] = None) -> float:
        """
        Calculate elapsed time in seconds.
        
        Args:
            start: Start datetime
            end: End datetime (defaults to now)
            
        Returns:
            Elapsed seconds
        """
        if end is None:
            end = DateTimeUtils.now()
        return (end - start).total_seconds()
    
    @staticmethod
    def elapsed_human(start: datetime, end: Optional[datetime] = None) -> str:
        """
        Get human-readable elapsed time.
        
        Args:
            start: Start datetime
            end: End datetime (defaults to now)
            
        Returns:
            Human-readable string (e.g., "2h 30m 15s")
        """
        seconds = int(DateTimeUtils.elapsed(start, end))
        
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs}s"
    
    @staticmethod
    def add_seconds(dt: datetime, seconds: float) -> datetime:
        """Add seconds to datetime."""
        return dt + timedelta(seconds=seconds)
    
    @staticmethod
    def add_minutes(dt: datetime, minutes: float) -> datetime:
        """Add minutes to datetime."""
        return dt + timedelta(minutes=minutes)
    
    @staticmethod
    def add_hours(dt: datetime, hours: float) -> datetime:
        """Add hours to datetime."""
        return dt + timedelta(hours=hours)
    
    @staticmethod
    def add_days(dt: datetime, days: float) -> datetime:
        """Add days to datetime."""
        return dt + timedelta(days=days)
    
    @staticmethod
    def is_expired(dt: datetime, ttl_seconds: float) -> bool:
        """
        Check if datetime is expired based on TTL.
        
        Args:
            dt: Datetime to check
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if expired
        """
        return DateTimeUtils.elapsed(dt) > ttl_seconds
    
    @staticmethod
    def expires_at(dt: datetime, ttl_seconds: float) -> datetime:
        """
        Calculate expiration datetime.
        
        Args:
            dt: Base datetime
            ttl_seconds: Time to live in seconds
            
        Returns:
            Expiration datetime
        """
        return DateTimeUtils.add_seconds(dt, ttl_seconds)


# Convenience functions
def now() -> datetime:
    """Get current datetime."""
    return DateTimeUtils.now()


def now_iso() -> str:
    """Get current datetime as ISO string."""
    return DateTimeUtils.now_iso()


def parse_iso(iso_string: str) -> datetime:
    """Parse ISO datetime string."""
    return DateTimeUtils.parse_iso(iso_string)


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string."""
    return DateTimeUtils.to_iso(dt)


def elapsed(start: datetime, end: Optional[datetime] = None) -> float:
    """Calculate elapsed seconds."""
    return DateTimeUtils.elapsed(start, end)




