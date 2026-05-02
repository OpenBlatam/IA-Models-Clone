"""
Date Utilities Module
Date/time handling utilities using python-dateutil and pytz.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning("python-dateutil not available. Using basic date parsing.")

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    logger.warning("pytz not available. Using UTC only.")


class DateUtils:
    """Date/time utility functions."""
    
    DEFAULT_TIMEZONE = "UTC"
    
    @staticmethod
    def now(timezone: str = None) -> datetime:
        """Get current datetime in specified timezone."""
        tz = DateUtils._get_timezone(timezone)
        if tz:
            return datetime.now(tz)
        return datetime.utcnow()
    
    @staticmethod
    def parse(date_string: str, timezone: str = None) -> Optional[datetime]:
        """Parse a date string into datetime."""
        try:
            if DATEUTIL_AVAILABLE:
                dt = date_parser.parse(date_string)
            else:
                # Basic ISO format parsing
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        dt = datetime.strptime(date_string, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
            
            if timezone:
                tz = DateUtils._get_timezone(timezone)
                if tz:
                    dt = dt.replace(tzinfo=tz)
            
            return dt
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_string}': {e}")
            return None
    
    @staticmethod
    def format(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string."""
        return dt.strftime(fmt)
    
    @staticmethod
    def to_iso(dt: datetime) -> str:
        """Convert datetime to ISO format."""
        return dt.isoformat()
    
    @staticmethod
    def from_timestamp(timestamp: float, timezone: str = None) -> datetime:
        """Create datetime from Unix timestamp."""
        dt = datetime.utcfromtimestamp(timestamp)
        if timezone:
            tz = DateUtils._get_timezone(timezone)
            if tz:
                dt = dt.replace(tzinfo=pytz.UTC).astimezone(tz)
        return dt
    
    @staticmethod
    def add(
        dt: datetime,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        weeks: int = 0,
        months: int = 0,
        years: int = 0
    ) -> datetime:
        """Add time to datetime."""
        if DATEUTIL_AVAILABLE and (months or years):
            return dt + relativedelta(
                years=years,
                months=months,
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds
            )
        return dt + timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
    
    @staticmethod
    def diff(dt1: datetime, dt2: datetime) -> timedelta:
        """Get difference between two datetimes."""
        return dt1 - dt2
    
    @staticmethod
    def diff_human(dt1: datetime, dt2: datetime = None) -> str:
        """Get human-readable time difference."""
        if dt2 is None:
            dt2 = DateUtils.now()
        
        delta = abs(dt1 - dt2)
        
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
    
    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """Convert datetime between timezones."""
        if not PYTZ_AVAILABLE:
            return dt
        
        from_timezone = DateUtils._get_timezone(from_tz)
        to_timezone = DateUtils._get_timezone(to_tz)
        
        if from_timezone and to_timezone:
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            return dt.astimezone(to_timezone)
        
        return dt
    
    @staticmethod
    def _get_timezone(tz_name: str) -> Optional["pytz.timezone"]:
        """Get timezone object."""
        if not PYTZ_AVAILABLE or not tz_name:
            return None
        try:
            return pytz.timezone(tz_name)
        except:
            return None
    
    @staticmethod
    def list_timezones() -> list:
        """List common timezones."""
        if PYTZ_AVAILABLE:
            return pytz.common_timezones
        return ["UTC"]
    
    @staticmethod
    def is_business_day(dt: datetime) -> bool:
        """Check if date is a business day (Mon-Fri)."""
        return dt.weekday() < 5
    
    @staticmethod
    def next_business_day(dt: datetime) -> datetime:
        """Get next business day."""
        next_day = dt + timedelta(days=1)
        while not DateUtils.is_business_day(next_day):
            next_day += timedelta(days=1)
        return next_day
