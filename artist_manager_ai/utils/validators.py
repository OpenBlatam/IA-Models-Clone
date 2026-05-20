"""
Advanced Validators
==================

Comprehensive data validation utilities.
Refactored with best practices and type safety.
"""

import logging
import re
from typing import Any, Optional, Tuple, List, Dict, Callable
from datetime import datetime, date, time
from email.utils import parseaddr
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class Validator:
    """
    Advanced data validator with comprehensive validation methods.
    
    Features:
    - Email validation
    - URL validation
    - Date/time validation
    - ID validation
    - Range validation
    - Custom validators
    """
    
    # Email regex pattern (RFC 5322 compliant)
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'^https?://[^\s/$.?#].[^\s]*$'
    )
    
    # Phone pattern (international format)
    PHONE_PATTERN = re.compile(
        r'^\+?[1-9]\d{1,14}$'
    )
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, Optional[str]]:
        """
        Validate email address.
        
        Args:
            email: Email address to validate
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not email or not isinstance(email, str):
            return False, "Email must be a non-empty string"
        
        email = email.strip()
        
        if not Validator.EMAIL_PATTERN.match(email):
            return False, "Invalid email format"
        
        # Additional check using email.utils
        name, addr = parseaddr(email)
        if not addr or '@' not in addr:
            return False, "Invalid email address"
        
        return True, None
    
    @staticmethod
    def validate_url(
        url: str,
        schemes: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate URL.
        
        Args:
            url: URL to validate
            schemes: Allowed schemes (default: ['http', 'https'])
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"
        
        url = url.strip()
        
        if schemes is None:
            schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                return False, "URL must include a scheme (http/https)"
            
            if parsed.scheme not in schemes:
                return False, f"URL scheme must be one of: {', '.join(schemes)}"
            
            if not parsed.netloc:
                return False, "URL must include a domain"
            
            return True, None
        except Exception as e:
            return False, f"Invalid URL: {str(e)}"
    
    @staticmethod
    def validate_phone(phone: str) -> Tuple[bool, Optional[str]]:
        """
        Validate phone number (international format).
        
        Args:
            phone: Phone number to validate
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not phone or not isinstance(phone, str):
            return False, "Phone must be a non-empty string"
        
        # Remove common separators
        phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
        
        if not Validator.PHONE_PATTERN.match(phone_clean):
            return False, "Invalid phone number format"
        
        return True, None
    
    @staticmethod
    def validate_artist_id(artist_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate artist ID.
        
        Args:
            artist_id: Artist ID to validate
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not artist_id or not isinstance(artist_id, str):
            return False, "Artist ID must be a non-empty string"
        
        artist_id = artist_id.strip()
        
        if len(artist_id) == 0:
            return False, "Artist ID cannot be empty"
        
        if len(artist_id) > 100:
            return False, "Artist ID cannot exceed 100 characters"
        
        # Allow alphanumeric, underscore, and hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', artist_id):
            return False, "Artist ID can only contain alphanumeric characters, underscore, and hyphen"
        
        return True, None
    
    @staticmethod
    def validate_event_time(
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate event time range.
        
        Args:
            start_time: Event start time
            end_time: Event end time
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(start_time, datetime):
            return False, "Start time must be a datetime object"
        
        if not isinstance(end_time, datetime):
            return False, "End time must be a datetime object"
        
        if end_time <= start_time:
            return False, "End time must be after start time"
        
        return True, None
    
    @staticmethod
    def validate_time_range(
        start_time: datetime,
        end_time: datetime,
        min_duration_minutes: int = 0,
        max_duration_hours: int = 24
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate time range with duration constraints.
        
        Args:
            start_time: Start time
            end_time: End time
            min_duration_minutes: Minimum duration in minutes
            max_duration_hours: Maximum duration in hours
        
        Returns:
            (is_valid, error_message) tuple
        """
        # Validate basic time range
        is_valid, error = Validator.validate_event_time(start_time, end_time)
        if not is_valid:
            return is_valid, error
        
        # Calculate duration
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        if duration_minutes < min_duration_minutes:
            return False, f"Duration must be at least {min_duration_minutes} minutes"
        
        if duration_minutes > max_duration_hours * 60:
            return False, f"Duration cannot exceed {max_duration_hours} hours"
        
        return True, None
    
    @staticmethod
    def validate_priority(priority: int) -> Tuple[bool, Optional[str]]:
        """
        Validate priority value.
        
        Args:
            priority: Priority value (1-10)
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(priority, int):
            return False, "Priority must be an integer"
        
        if not (1 <= priority <= 10):
            return False, "Priority must be between 1 and 10"
        
        return True, None
    
    @staticmethod
    def validate_days_of_week(days: List[int]) -> Tuple[bool, Optional[str]]:
        """
        Validate days of week.
        
        Args:
            days: List of day numbers (0=Monday, 6=Sunday)
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(days, list):
            return False, "Days must be a list"
        
        if len(days) == 0:
            return False, "Days list cannot be empty"
        
        for day in days:
            if not isinstance(day, int):
                return False, "All days must be integers"
            if not (0 <= day <= 6):
                return False, f"Day {day} must be between 0 and 6"
        
        # Check for duplicates
        if len(days) != len(set(days)):
            return False, "Days list contains duplicates"
        
        return True, None
    
    @staticmethod
    def validate_string(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate string with constraints.
        
        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            pattern: Regex pattern
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if min_length is not None and len(value) < min_length:
            return False, f"String must be at least {min_length} characters"
        
        if max_length is not None and len(value) > max_length:
            return False, f"String cannot exceed {max_length} characters"
        
        if pattern and not re.match(pattern, value):
            return False, f"String does not match required pattern"
        
        return True, None
    
    @staticmethod
    def validate_number(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate number with range constraints.
        
        Args:
            value: Number to validate
            min_value: Minimum value
            max_value: Maximum value
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(value, (int, float)):
            return False, "Value must be a number"
        
        if min_value is not None and value < min_value:
            return False, f"Value must be at least {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value cannot exceed {max_value}"
        
        return True, None
    
    @staticmethod
    def validate_dict(
        data: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate dictionary structure.
        
        Args:
            data: Dictionary to validate
            required_keys: Required keys
            allowed_keys: Allowed keys (None = all keys allowed)
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(data, dict):
            return False, "Value must be a dictionary"
        
        if required_keys:
            missing = [key for key in required_keys if key not in data]
            if missing:
                return False, f"Missing required keys: {', '.join(missing)}"
        
        if allowed_keys:
            invalid = [key for key in data.keys() if key not in allowed_keys]
            if invalid:
                return False, f"Invalid keys: {', '.join(invalid)}"
        
        return True, None
    
    @staticmethod
    def validate_all(
        validators: List[Callable[[], Tuple[bool, Optional[str]]]]
    ) -> Tuple[bool, List[str]]:
        """
        Run multiple validators and collect all errors.
        
        Args:
            validators: List of validator functions
        
        Returns:
            (all_valid, error_messages) tuple
        """
        errors = []
        for validator in validators:
            is_valid, error = validator()
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
