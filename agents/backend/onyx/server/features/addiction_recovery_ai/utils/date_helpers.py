"""
Date and time helper utilities
"""

from datetime import datetime, timedelta, timezone
from typing import Optional


def get_current_utc() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


def parse_iso_date(date_str: str) -> datetime:
    """
    Parse ISO format date string
    
    Args:
        date_str: ISO format date string
    
    Returns:
        Parsed datetime
    
    Raises:
        ValueError if date format is invalid
    """
    try:
        # Handle Z suffix for UTC
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str)
    except ValueError as e:
        raise ValueError(f"Invalid ISO date format: {date_str}") from e


def format_iso_date(dt: datetime) -> str:
    """
    Format datetime to ISO string
    
    Args:
        dt: Datetime object
    
    Returns:
        ISO format string
    """
    return dt.isoformat()


def add_days(dt: datetime, days: int) -> datetime:
    """Add days to datetime"""
    return dt + timedelta(days=days)


def subtract_days(dt: datetime, days: int) -> datetime:
    """Subtract days from datetime"""
    return dt - timedelta(days=days)


def days_between(start: datetime, end: datetime) -> int:
    """Calculate days between two datetimes"""
    return (end - start).days


def is_within_range(
    dt: datetime,
    start: datetime,
    end: datetime
) -> bool:
    """Check if datetime is within range"""
    return start <= dt <= end


def get_start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00)"""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59)"""
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_week(dt: datetime) -> datetime:
    """Get start of week (Monday)"""
    days_since_monday = dt.weekday()
    return get_start_of_day(dt - timedelta(days=days_since_monday))


def get_start_of_month(dt: datetime) -> datetime:
    """Get start of month"""
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

