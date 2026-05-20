"""
DateTime Helper Functions

Helper functions for common datetime calculations and operations that appear
throughout the codebase to improve consistency and reduce duplication.
"""

from datetime import datetime, timedelta
from typing import Optional
from ..constants import SECONDS_PER_HOUR


def calculate_age_hours(created_at: datetime, reference_time: Optional[datetime] = None) -> float:
    """
    Calculate age in hours from a datetime to a reference time (or now).
    
    This helper encapsulates the common pattern of calculating age in hours
    that appears in ranking and trending calculations.
    
    Args:
        created_at: Creation datetime
        reference_time: Reference time (default: current UTC time)
        
    Returns:
        Age in hours as float
        
    Raises:
        TypeError: If created_at is not a datetime object
        
    Example:
        >>> age_hours = calculate_age_hours(chat.created_at)
        >>> age_hours = calculate_age_hours(chat.created_at, reference_time=now)
    """
    if not isinstance(created_at, datetime):
        raise TypeError(f"created_at must be a datetime object, got {type(created_at).__name__}")
    
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    if not isinstance(reference_time, datetime):
        raise TypeError(f"reference_time must be a datetime object, got {type(reference_time).__name__}")
    
    time_diff = reference_time - created_at
    return time_diff.total_seconds() / SECONDS_PER_HOUR


def calculate_cutoff_time(hours: int, reference_time: Optional[datetime] = None) -> datetime:
    """
    Calculate cutoff time by subtracting hours from reference time (or now).
    
    This helper encapsulates the common pattern of calculating cutoff times
    for trending and time-based queries.
    
    Args:
        hours: Number of hours to subtract (must be >= 0)
        reference_time: Reference time (default: current UTC time)
        
    Returns:
        Cutoff datetime
        
    Raises:
        ValueError: If hours is negative
        
    Example:
        >>> cutoff_time = calculate_cutoff_time(24)  # 24 hours ago
        >>> cutoff_time = calculate_cutoff_time(hours, reference_time=now)
    """
    if hours < 0:
        raise ValueError(f"hours must be >= 0, got {hours}")
    
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    if not isinstance(reference_time, datetime):
        raise TypeError(f"reference_time must be a datetime object, got {type(reference_time).__name__}")
    
    return reference_time - timedelta(hours=hours)


def calculate_cutoff_time_days(days: int, reference_time: Optional[datetime] = None) -> datetime:
    """
    Calculate cutoff time by subtracting days from reference time (or now).
    
    This helper encapsulates the common pattern of calculating cutoff times
    for date-based queries and analytics.
    
    Args:
        days: Number of days to subtract (must be >= 0)
        reference_time: Reference time (default: current UTC time)
        
    Returns:
        Cutoff datetime
        
    Raises:
        ValueError: If days is negative
        
    Example:
        >>> cutoff_time = calculate_cutoff_time_days(7)  # 7 days ago
    """
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")
    
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    if not isinstance(reference_time, datetime):
        raise TypeError(f"reference_time must be a datetime object, got {type(reference_time).__name__}")
    
    return reference_time - timedelta(days=days)


def format_datetime_iso(dt: Optional[datetime] = None) -> str:
    """
    Format datetime to ISO format string.
    
    This helper encapsulates the common pattern of formatting datetimes
    to ISO format that appears in API responses.
    
    Args:
        dt: Datetime to format (default: current UTC time)
        
    Returns:
        ISO format string (e.g., "2024-01-01T12:00:00")
        
    Example:
        >>> timestamp = format_datetime_iso()
        >>> timestamp = format_datetime_iso(datetime.utcnow())
    """
    if dt is None:
        dt = datetime.utcnow()
    
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    
    return dt.isoformat()


def is_within_time_window(
    created_at: datetime,
    hours: int,
    reference_time: Optional[datetime] = None
) -> bool:
    """
    Check if a datetime is within a time window (hours from reference time).
    
    This helper encapsulates the common pattern of checking if something
    is within a time window for trending calculations.
    
    Args:
        created_at: Creation datetime to check
        hours: Time window in hours
        reference_time: Reference time (default: current UTC time)
        
    Returns:
        True if created_at is within the time window, False otherwise
        
    Example:
        >>> if is_within_time_window(chat.created_at, 24):
        >>>     # chat is within last 24 hours
    """
    if hours < 0:
        return False
    
    age_hours = calculate_age_hours(created_at, reference_time)
    return age_hours <= hours

