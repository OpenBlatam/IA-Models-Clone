"""
Time Testing Helpers
Specialized helpers for time and date testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
import time


class TimeTestHelpers:
    """Helpers for time testing"""
    
    @staticmethod
    def create_mock_time_provider(
        current_time: Optional[datetime] = None
    ) -> Mock:
        """Create mock time provider"""
        time_provider = Mock()
        current = current_time or datetime.utcnow()
        time_provider.now = Mock(return_value=current)
        time_provider.utcnow = Mock(return_value=current)
        time_provider.timestamp = Mock(return_value=current.timestamp())
        return time_provider
    
    @staticmethod
    def freeze_time(target_time: datetime):
        """Context manager to freeze time"""
        class TimeFreezer:
            def __init__(self, time_to_freeze):
                self.time_to_freeze = time_to_freeze
                self.patches = []
            
            def __enter__(self):
                # Patch datetime.utcnow
                self.patches.append(
                    patch('datetime.datetime.utcnow', return_value=self.time_to_freeze)
                )
                # Patch time.time if needed
                self.patches.append(
                    patch('time.time', return_value=self.time_to_freeze.timestamp())
                )
                for p in self.patches:
                    p.start()
                return self
            
            def __exit__(self, *args):
                for p in self.patches:
                    p.stop()
        
        return TimeFreezer(target_time)
    
    @staticmethod
    def assert_time_approx(
        actual: datetime,
        expected: datetime,
        tolerance_seconds: int = 5
    ):
        """Assert time is approximately equal"""
        diff = abs((actual - expected).total_seconds())
        assert diff <= tolerance_seconds, \
            f"Time difference {diff}s exceeds tolerance {tolerance_seconds}s"
    
    @staticmethod
    def assert_time_before(earlier: datetime, later: datetime):
        """Assert one time is before another"""
        assert earlier < later, \
            f"Time {earlier} is not before {later}"
    
    @staticmethod
    def assert_time_after(later: datetime, earlier: datetime):
        """Assert one time is after another"""
        assert later > earlier, \
            f"Time {later} is not after {earlier}"


class DateRangeHelpers:
    """Helpers for date range testing"""
    
    @staticmethod
    def create_date_range(
        start_date: datetime,
        end_date: datetime,
        interval: timedelta = timedelta(days=1)
    ) -> List[datetime]:
        """Create list of dates in range"""
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += interval
        return dates
    
    @staticmethod
    def assert_date_in_range(
        date: datetime,
        start_date: datetime,
        end_date: datetime
    ):
        """Assert date is within range"""
        assert start_date <= date <= end_date, \
            f"Date {date} is not in range [{start_date}, {end_date}]"
    
    @staticmethod
    def assert_dates_ordered(dates: List[datetime]):
        """Assert dates are in chronological order"""
        for i in range(len(dates) - 1):
            assert dates[i] <= dates[i + 1], \
                f"Dates are not in order: {dates[i]} > {dates[i + 1]}"


class TimezoneHelpers:
    """Helpers for timezone testing"""
    
    @staticmethod
    def convert_to_utc(dt: datetime, tz_name: str = "UTC") -> datetime:
        """Convert datetime to UTC"""
        if dt.tzinfo is None:
            # Assume naive datetime is in specified timezone
            import pytz
            tz = pytz.timezone(tz_name)
            dt = tz.localize(dt)
        return dt.astimezone(timezone.utc)
    
    @staticmethod
    def assert_timezone(dt: datetime, expected_tz: timezone):
        """Assert datetime has expected timezone"""
        assert dt.tzinfo == expected_tz, \
            f"Datetime timezone {dt.tzinfo} does not match expected {expected_tz}"


class DurationHelpers:
    """Helpers for duration testing"""
    
    @staticmethod
    def assert_duration_within_range(
        duration: timedelta,
        min_duration: timedelta,
        max_duration: timedelta
    ):
        """Assert duration is within range"""
        assert min_duration <= duration <= max_duration, \
            f"Duration {duration} is not in range [{min_duration}, {max_duration}]"
    
    @staticmethod
    def assert_duration_less_than(duration: timedelta, max_duration: timedelta):
        """Assert duration is less than maximum"""
        assert duration <= max_duration, \
            f"Duration {duration} exceeds maximum {max_duration}"
    
    @staticmethod
    def assert_duration_greater_than(duration: timedelta, min_duration: timedelta):
        """Assert duration is greater than minimum"""
        assert duration >= min_duration, \
            f"Duration {duration} is less than minimum {min_duration}"


# Convenience exports
create_mock_time_provider = TimeTestHelpers.create_mock_time_provider
freeze_time = TimeTestHelpers.freeze_time
assert_time_approx = TimeTestHelpers.assert_time_approx
assert_time_before = TimeTestHelpers.assert_time_before
assert_time_after = TimeTestHelpers.assert_time_after

create_date_range = DateRangeHelpers.create_date_range
assert_date_in_range = DateRangeHelpers.assert_date_in_range
assert_dates_ordered = DateRangeHelpers.assert_dates_ordered

convert_to_utc = TimezoneHelpers.convert_to_utc
assert_timezone = TimezoneHelpers.assert_timezone

assert_duration_within_range = DurationHelpers.assert_duration_within_range
assert_duration_less_than = DurationHelpers.assert_duration_less_than
assert_duration_greater_than = DurationHelpers.assert_duration_greater_than



