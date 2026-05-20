"""
Time Utilities
==============

Utilities for time and date operations.
"""

import time
from typing import Optional
from datetime import datetime, timedelta, timezone


def get_utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def get_timestamp() -> float:
    """
    Get current Unix timestamp.
    
    Returns:
        Unix timestamp
    """
    return time.time()


def format_datetime(dt: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        format: Format string
        
    Returns:
        Formatted string
    """
    return dt.strftime(format)


def parse_datetime(dt_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse datetime from string.
    
    Args:
        dt_str: Datetime string
        format: Format string
        
    Returns:
        Datetime object
    """
    return datetime.strptime(dt_str, format)


def add_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """
    Add time to datetime.
    
    Args:
        dt: Datetime object
        days: Days to add
        hours: Hours to add
        minutes: Minutes to add
        seconds: Seconds to add
        
    Returns:
        New datetime object
    """
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return dt + delta


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Args:
        dt: Past datetime
        
    Returns:
        Human-readable string (e.g., "2 hours ago")
    """
    now = datetime.now(dt.tzinfo if dt.tzinfo else None)
    delta = now - dt
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def is_expired(dt: datetime, ttl_seconds: float) -> bool:
    """
    Check if datetime is expired based on TTL.
    
    Args:
        dt: Datetime to check
        ttl_seconds: Time to live in seconds
        
    Returns:
        True if expired, False otherwise
    """
    now = datetime.now(dt.tzinfo if dt.tzinfo else None)
    return (now - dt).total_seconds() > ttl_seconds


def sleep_until(target_time: datetime):
    """
    Sleep until target time.
    
    Args:
        target_time: Target datetime
    """
    now = datetime.now(target_time.tzinfo if target_time.tzinfo else None)
    if target_time > now:
        delta = (target_time - now).total_seconds()
        time.sleep(delta)

