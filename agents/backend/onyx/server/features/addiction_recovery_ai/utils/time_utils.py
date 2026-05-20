"""
Time utilities
Time and date manipulation functions
"""

from typing import Optional, List
from datetime import datetime, timedelta, timezone
import time


def get_timestamp() -> float:
    """
    Get current timestamp
    
    Returns:
        Current timestamp
    """
    return time.time()


def get_utc_now() -> datetime:
    """
    Get current UTC datetime
    
    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def get_local_now() -> datetime:
    """
    Get current local datetime
    
    Returns:
        Current local datetime
    """
    return datetime.now()


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to timestamp
    
    Args:
        dt: Datetime object
    
    Returns:
        Timestamp
    """
    return dt.timestamp()


def timestamp_to_datetime(ts: float, tz: Optional[timezone] = None) -> datetime:
    """
    Convert timestamp to datetime
    
    Args:
        ts: Timestamp
        tz: Optional timezone
    
    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(ts, tz=tz)


def add_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """
    Add time to datetime
    
    Args:
        dt: Datetime object
        days: Days to add
        hours: Hours to add
        minutes: Minutes to add
        seconds: Seconds to add
    
    Returns:
        New datetime
    """
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return dt + delta


def subtract_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """
    Subtract time from datetime
    
    Args:
        dt: Datetime object
        days: Days to subtract
        hours: Hours to subtract
        minutes: Minutes to subtract
        seconds: Seconds to subtract
    
    Returns:
        New datetime
    """
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return dt - delta


def time_difference(start: datetime, end: datetime) -> timedelta:
    """
    Calculate time difference
    
    Args:
        start: Start datetime
        end: End datetime
    
    Returns:
        Time difference
    """
    return end - start


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    
    if seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f}h"
    
    days = seconds / 86400
    return f"{days:.2f}d"


def is_business_day(dt: datetime) -> bool:
    """
    Check if datetime is business day (Monday-Friday)
    
    Args:
        dt: Datetime object
    
    Returns:
        True if business day
    """
    return dt.weekday() < 5


def get_business_days(start: datetime, end: datetime) -> int:
    """
    Get number of business days between dates
    
    Args:
        start: Start datetime
        end: End datetime
    
    Returns:
        Number of business days
    """
    count = 0
    current = start
    
    while current <= end:
        if is_business_day(current):
            count += 1
        current += timedelta(days=1)
    
    return count


def sleep_seconds(seconds: float) -> None:
    """
    Sleep for specified seconds
    
    Args:
        seconds: Seconds to sleep
    """
    time.sleep(seconds)


def sleep_milliseconds(ms: int) -> None:
    """
    Sleep for specified milliseconds
    
    Args:
        ms: Milliseconds to sleep
    """
    time.sleep(ms / 1000.0)


def measure_time(func):
    """
    Decorator to measure function execution time
    
    Returns:
        Decorator function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def get_timezone_offset(tz: timezone) -> timedelta:
    """
    Get timezone offset
    
    Args:
        tz: Timezone object
    
    Returns:
        Timezone offset
    """
    return tz.utcoffset(datetime.now())


def convert_timezone(dt: datetime, tz: timezone) -> datetime:
    """
    Convert datetime to timezone
    
    Args:
        dt: Datetime object
        tz: Target timezone
    
    Returns:
        Converted datetime
    """
    return dt.astimezone(tz)

