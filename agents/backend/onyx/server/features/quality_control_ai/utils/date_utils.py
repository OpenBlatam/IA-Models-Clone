"""
Date Utilities

Utility functions for date and time operations.
"""

from datetime import datetime, timedelta
from typing import Optional
import pytz


def now_utc() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current UTC datetime
    """
    return datetime.utcnow()


def now_local() -> datetime:
    """
    Get current local datetime.
    
    Returns:
        Current local datetime
    """
    return datetime.now()


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.
    
    Args:
        dt: Datetime to convert
    
    Returns:
        UTC datetime
    """
    if dt.tzinfo is None:
        # Assume local time
        return dt.replace(tzinfo=pytz.UTC)
    return dt.astimezone(pytz.UTC)


def format_datetime(
    dt: datetime,
    format: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime to format
        format: Format string
    
    Returns:
        Formatted string
    """
    return dt.strftime(format)


def parse_datetime(
    date_string: str,
    format: str = "%Y-%m-%d %H:%M:%S"
) -> Optional[datetime]:
    """
    Parse datetime from string.
    
    Args:
        date_string: Date string
        format: Format string
    
    Returns:
        Parsed datetime or None
    """
    try:
        return datetime.strptime(date_string, format)
    except ValueError:
        return None


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Args:
        dt: Past datetime
    
    Returns:
        Time ago string (e.g., "2 hours ago")
    """
    now = datetime.utcnow()
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


def is_within_timeframe(
    dt: datetime,
    timeframe: timedelta
) -> bool:
    """
    Check if datetime is within timeframe from now.
    
    Args:
        dt: Datetime to check
        timeframe: Timeframe (e.g., timedelta(hours=1))
    
    Returns:
        True if within timeframe
    """
    now = datetime.utcnow()
    return (now - dt) <= timeframe



