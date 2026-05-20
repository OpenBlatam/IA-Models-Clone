"""
Date and time helper functions.

This module provides utilities for date/time formatting, parsing,
and manipulation with consistent patterns.
"""

from typing import Optional, Union
from datetime import datetime, timedelta
import time


def format_timestamp(
    dt: Optional[datetime] = None,
    format: str = "iso",
    timezone: Optional[str] = None
) -> str:
    """
    Format datetime to string in various formats.
    
    Args:
        dt: Datetime object (uses current time if None)
        format: Format type - "iso", "rfc3339", "unix", or strftime format
        timezone: Optional timezone (not implemented, placeholder)
    
    Returns:
        Formatted datetime string
    
    Example:
        iso = format_timestamp()  # "2024-01-01T12:00:00"
        iso = format_timestamp(dt, format="iso")
        custom = format_timestamp(dt, format="%Y-%m-%d %H:%M:%S")
    """
    if dt is None:
        dt = datetime.utcnow()
    
    if format == "iso":
        return dt.isoformat()
    elif format == "rfc3339":
        return dt.isoformat() + "Z"
    elif format == "unix":
        return str(int(dt.timestamp()))
    else:
        # Custom strftime format
        return dt.strftime(format)


def parse_timestamp(
    value: Union[str, int, float],
    format: Optional[str] = None
) -> datetime:
    """
    Parse timestamp from various formats.
    
    Args:
        value: Timestamp value (string, int, or float)
        format: Optional strftime format for string parsing
    
    Returns:
        Datetime object
    
    Example:
        dt = parse_timestamp("2024-01-01T12:00:00")
        dt = parse_timestamp(1704110400)  # Unix timestamp
        dt = parse_timestamp("2024-01-01", format="%Y-%m-%d")
    """
    if isinstance(value, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(value)
    
    if isinstance(value, str):
        if format:
            return datetime.strptime(value, format)
        else:
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                # Try common formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse timestamp: {value}")
    
    raise ValueError(f"Invalid timestamp type: {type(value)}")


def get_current_timestamp(format: str = "iso") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format: Format type (see format_timestamp)
    
    Returns:
        Formatted current timestamp
    
    Example:
        now = get_current_timestamp()  # ISO format
        now = get_current_timestamp(format="unix")  # Unix timestamp
    """
    return format_timestamp(datetime.utcnow(), format=format)


def format_duration(
    seconds: Union[int, float],
    format: str = "human"
) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        format: Format type - "human", "iso8601", or "compact"
    
    Returns:
        Formatted duration string
    
    Example:
        human = format_duration(3661)  # "1h 1m 1s"
        iso = format_duration(3661, format="iso8601")  # "PT1H1M1S"
        compact = format_duration(3661, format="compact")  # "1:01:01"
    """
    if format == "human":
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    elif format == "iso8601":
        # ISO 8601 duration format
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = ["PT"]
        if hours > 0:
            parts.append(f"{hours}H")
        if minutes > 0:
            parts.append(f"{minutes}M")
        if secs > 0:
            parts.append(f"{secs}S")
        
        return "".join(parts)
    
    elif format == "compact":
        # Compact format like "1:01:01"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    return str(seconds)


def time_ago(dt: datetime) -> str:
    """
    Get human-readable "time ago" string.
    
    Args:
        dt: Past datetime
    
    Returns:
        "Time ago" string
    
    Example:
        ago = time_ago(datetime.now() - timedelta(hours=2))  # "2 hours ago"
        ago = time_ago(datetime.now() - timedelta(days=5))  # "5 days ago"
    """
    now = datetime.utcnow()
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








