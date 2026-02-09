"""Time utilities."""

import time
from datetime import datetime, timedelta
from typing import Optional


def get_timestamp() -> float:
    """
    Get current Unix timestamp.
    
    Returns:
        Unix timestamp
    """
    return time.time()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        Unix timestamp
    """
    return dt.timestamp()


def sleep_seconds(seconds: float) -> None:
    """
    Sleep for specified seconds.
    
    Args:
        seconds: Seconds to sleep
    """
    time.sleep(seconds)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours < 24:
        return f"{hours}h {mins}m {secs}s"
    
    days = int(hours // 24)
    hrs = int(hours % 24)
    
    return f"{days}d {hrs}h {mins}m"


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., "1h 30m", "45s")
        
    Returns:
        Duration in seconds
    """
    import re
    
    pattern = r'(\d+)([dhms])'
    matches = re.findall(pattern, duration_str.lower())
    
    total_seconds = 0.0
    
    multipliers = {
        'd': 86400,
        'h': 3600,
        'm': 60,
        's': 1
    }
    
    for value, unit in matches:
        total_seconds += float(value) * multipliers.get(unit, 0)
    
    return total_seconds


def get_timezone_offset() -> timedelta:
    """
    Get local timezone offset from UTC.
    
    Returns:
        Timezone offset
    """
    import time
    offset = time.timezone if (time.daylight == 0) else time.altzone
    return timedelta(seconds=-offset)


def is_weekend(dt: datetime) -> bool:
    """
    Check if date is weekend.
    
    Args:
        dt: Datetime object
        
    Returns:
        True if weekend
    """
    return dt.weekday() >= 5


def is_weekday(dt: datetime) -> bool:
    """
    Check if date is weekday.
    
    Args:
        dt: Datetime object
        
    Returns:
        True if weekday
    """
    return dt.weekday() < 5


def get_business_days(start: datetime, end: datetime) -> int:
    """
    Get number of business days between dates.
    
    Args:
        start: Start date
        end: End date
        
    Returns:
        Number of business days
    """
    business_days = 0
    current = start
    
    while current <= end:
        if is_weekday(current):
            business_days += 1
        current += timedelta(days=1)
    
    return business_days

