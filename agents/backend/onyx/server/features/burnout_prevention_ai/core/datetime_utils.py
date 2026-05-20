"""
DateTime Utilities
==================
Utility functions for datetime operations.

Centralizes UTC datetime handling to ensure consistency
across the application and avoid timezone-related bugs.
"""

from datetime import datetime, timezone
from typing import Optional


def get_utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current datetime in UTC
    """
    return datetime.now(timezone.utc)


def get_iso_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Get ISO format timestamp.
    
    Args:
        dt: Optional datetime. If None, uses current time.
        
    Returns:
        ISO format timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def get_utc_iso_timestamp() -> str:
    """
    Get UTC ISO format timestamp.
    
    Returns:
        UTC ISO format timestamp string
    """
    return get_utc_now().isoformat()

