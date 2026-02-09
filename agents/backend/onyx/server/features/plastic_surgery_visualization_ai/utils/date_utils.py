"""Date and time utilities."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import pytz


def now_utc() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current UTC datetime
    """
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def now_local(tz: Optional[str] = None) -> datetime:
    """
    Get current local datetime.
    
    Args:
        tz: Timezone name (e.g., 'America/New_York')
        
    Returns:
        Current local datetime
    """
    if tz:
        tz_obj = pytz.timezone(tz)
        return datetime.now(tz_obj)
    return datetime.now()


def parse_datetime(date_string: str, format: Optional[str] = None) -> datetime:
    """
    Parse datetime string.
    
    Args:
        date_string: Date string to parse
        format: Optional format string
        
    Returns:
        Parsed datetime
    """
    if format:
        return datetime.strptime(date_string, format)
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {date_string}")


def format_datetime(dt: datetime, format: str = "%Y-%m-%dT%H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime to format
        format: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format)


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Args:
        dt: Datetime to compare
        
    Returns:
        Human-readable string (e.g., "2 hours ago")
    """
    now = now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    delta = now - dt
    
    if delta.total_seconds() < 60:
        return "just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days < 30:
        days = delta.days
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif delta.days < 365:
        months = int(delta.days / 30)
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = int(delta.days / 365)
        return f"{years} year{'s' if years != 1 else ''} ago"


def is_expired(dt: datetime, ttl_seconds: float) -> bool:
    """
    Check if datetime is expired based on TTL.
    
    Args:
        dt: Datetime to check
        ttl_seconds: Time to live in seconds
        
    Returns:
        True if expired
    """
    now = now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    elapsed = (now - dt).total_seconds()
    return elapsed > ttl_seconds


def add_time(dt: datetime, **kwargs) -> datetime:
    """
    Add time to datetime.
    
    Args:
        dt: Datetime
        **kwargs: Time components (days, hours, minutes, seconds)
        
    Returns:
        New datetime
    """
    delta = timedelta(**kwargs)
    return dt + delta


def get_start_of_day(dt: datetime) -> datetime:
    """
    Get start of day for datetime.
    
    Args:
        dt: Datetime
        
    Returns:
        Datetime at start of day
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime) -> datetime:
    """
    Get end of day for datetime.
    
    Args:
        dt: Datetime
        
    Returns:
        Datetime at end of day
    """
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_week(dt: datetime) -> datetime:
    """
    Get start of week (Monday) for datetime.
    
    Args:
        dt: Datetime
        
    Returns:
        Datetime at start of week
    """
    days_since_monday = dt.weekday()
    return get_start_of_day(dt - timedelta(days=days_since_monday))


def get_start_of_month(dt: datetime) -> datetime:
    """
    Get start of month for datetime.
    
    Args:
        dt: Datetime
        
    Returns:
        Datetime at start of month
    """
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

