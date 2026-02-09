"""
Date and time utilities.

Consolidates common date/time manipulation patterns.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Try to import dateutil, fallback to basic parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    logger.warning("python-dateutil not available, using basic date parsing")


class DateTimeUtils:
    """Utilities for date and time operations."""
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def now_iso() -> str:
        """Get current UTC datetime as ISO string."""
        return DateTimeUtils.now().isoformat()
    
    @staticmethod
    def parse(
        date_string: str,
        default: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Parse date string to datetime.
        
        Args:
            date_string: Date string to parse
            default: Default value if parsing fails
        
        Returns:
            Parsed datetime or default
        """
        try:
            if HAS_DATEUTIL:
                return date_parser.parse(date_string)
            else:
                # Basic ISO format parsing
                return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date '{date_string}': {e}")
            return default
    
    @staticmethod
    def format(
        dt: datetime,
        format_string: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """
        Format datetime to string.
        
        Args:
            dt: Datetime to format
            format_string: Format string
        
        Returns:
            Formatted string
        """
        return dt.strftime(format_string)
    
    @staticmethod
    def format_iso(dt: datetime) -> str:
        """Format datetime to ISO string."""
        return dt.isoformat()
    
    @staticmethod
    def add_days(
        dt: datetime,
        days: int
    ) -> datetime:
        """Add days to datetime."""
        return dt + timedelta(days=days)
    
    @staticmethod
    def add_hours(
        dt: datetime,
        hours: int
    ) -> datetime:
        """Add hours to datetime."""
        return dt + timedelta(hours=hours)
    
    @staticmethod
    def add_minutes(
        dt: datetime,
        minutes: int
    ) -> datetime:
        """Add minutes to datetime."""
        return dt + timedelta(minutes=minutes)
    
    @staticmethod
    def subtract_days(
        dt: datetime,
        days: int
    ) -> datetime:
        """Subtract days from datetime."""
        return dt - timedelta(days=days)
    
    @staticmethod
    def subtract_hours(
        dt: datetime,
        hours: int
    ) -> datetime:
        """Subtract hours from datetime."""
        return dt - timedelta(hours=hours)
    
    @staticmethod
    def subtract_minutes(
        dt: datetime,
        minutes: int
    ) -> datetime:
        """Subtract minutes from datetime."""
        return dt - timedelta(minutes=minutes)
    
    @staticmethod
    def time_ago(
        dt: datetime,
        now: Optional[datetime] = None
    ) -> timedelta:
        """
        Calculate time difference from now.
        
        Args:
            dt: Past datetime
            now: Current datetime (defaults to now)
        
        Returns:
            Time difference
        """
        if now is None:
            now = DateTimeUtils.now()
        return now - dt
    
    @staticmethod
    def time_until(
        dt: datetime,
        now: Optional[datetime] = None
    ) -> timedelta:
        """
        Calculate time difference until future datetime.
        
        Args:
            dt: Future datetime
            now: Current datetime (defaults to now)
        
        Returns:
            Time difference
        """
        if now is None:
            now = DateTimeUtils.now()
        return dt - now
    
    @staticmethod
    def is_past(
        dt: datetime,
        now: Optional[datetime] = None
    ) -> bool:
        """Check if datetime is in the past."""
        if now is None:
            now = DateTimeUtils.now()
        return dt < now
    
    @staticmethod
    def is_future(
        dt: datetime,
        now: Optional[datetime] = None
    ) -> bool:
        """Check if datetime is in the future."""
        if now is None:
            now = DateTimeUtils.now()
        return dt > now
    
    @staticmethod
    def is_within(
        dt: datetime,
        start: datetime,
        end: datetime
    ) -> bool:
        """Check if datetime is within range."""
        return start <= dt <= end
    
    @staticmethod
    def to_timestamp(dt: datetime) -> float:
        """Convert datetime to Unix timestamp."""
        return dt.timestamp()
    
    @staticmethod
    def from_timestamp(timestamp: float) -> datetime:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def start_of_day(dt: datetime) -> datetime:
        """Get start of day (00:00:00)."""
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    @staticmethod
    def end_of_day(dt: datetime) -> datetime:
        """Get end of day (23:59:59.999999)."""
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


# Convenience functions
def now() -> datetime:
    """Get current UTC datetime."""
    return DateTimeUtils.now()


def now_iso() -> str:
    """Get current UTC datetime as ISO string."""
    return DateTimeUtils.now_iso()


def parse_date(
    date_string: str,
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """Parse date string to datetime."""
    return DateTimeUtils.parse(date_string, default)


def format_date(
    dt: datetime,
    format_string: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format datetime to string."""
    return DateTimeUtils.format(dt, format_string)


def format_iso(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return DateTimeUtils.format_iso(dt)


def add_days(dt: datetime, days: int) -> datetime:
    """Add days to datetime."""
    return DateTimeUtils.add_days(dt, days)


def time_ago(dt: datetime, now: Optional[datetime] = None) -> timedelta:
    """Calculate time difference from now."""
    return DateTimeUtils.time_ago(dt, now)


def is_past(dt: datetime, now: Optional[datetime] = None) -> bool:
    """Check if datetime is in the past."""
    return DateTimeUtils.is_past(dt, now)


def is_future(dt: datetime, now: Optional[datetime] = None) -> bool:
    """Check if datetime is in the future."""
    return DateTimeUtils.is_future(dt, now)

