"""
Unified validation system for Suno Clone AI.

Consolidates all validation patterns into a single, consistent system.
"""

import re
import uuid
from typing import Any, Optional, List, Callable, Dict
from pathlib import Path
from datetime import datetime

from api.exceptions import InvalidInputError


class UnifiedValidator:
    """Unified validator with consistent validation patterns."""
    
    # Common patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    @staticmethod
    def validate_uuid(value: str, raise_on_error: bool = True) -> bool:
        """
        Validate UUID format.
        
        Args:
            value: UUID string to validate
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not value or not isinstance(value, str):
            if raise_on_error:
                raise InvalidInputError("UUID must be a non-empty string")
            return False
        
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            if raise_on_error:
                raise InvalidInputError(f"Invalid UUID format: {value}")
            return False
    
    @staticmethod
    def validate_email(email: str, raise_on_error: bool = True) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email string to validate
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not email or not isinstance(email, str):
            if raise_on_error:
                raise InvalidInputError("Email must be a non-empty string")
            return False
        
        if not UnifiedValidator.EMAIL_PATTERN.match(email):
            if raise_on_error:
                raise InvalidInputError(f"Invalid email format: {email}")
            return False
        
        return True
    
    @staticmethod
    def validate_url(url: str, raise_on_error: bool = True) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL string to validate
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not url or not isinstance(url, str):
            if raise_on_error:
                raise InvalidInputError("URL must be a non-empty string")
            return False
        
        if not UnifiedValidator.URL_PATTERN.match(url):
            if raise_on_error:
                raise InvalidInputError(f"Invalid URL format: {url}")
            return False
        
        return True
    
    @staticmethod
    def validate_string(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate string with length constraints.
        
        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not isinstance(value, str):
            if raise_on_error:
                raise InvalidInputError("Value must be a string")
            return False
        
        length = len(value.strip())
        
        if min_length is not None and length < min_length:
            if raise_on_error:
                raise InvalidInputError(f"String must be at least {min_length} characters")
            return False
        
        if max_length is not None and length > max_length:
            if raise_on_error:
                raise InvalidInputError(f"String must be at most {max_length} characters")
            return False
        
        return True
    
    @staticmethod
    def validate_number(
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate number with range constraints.
        
        Args:
            value: Number to validate
            min_value: Minimum value
            max_value: Maximum value
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not isinstance(value, (int, float)):
            if raise_on_error:
                raise InvalidInputError("Value must be a number")
            return False
        
        if min_value is not None and value < min_value:
            if raise_on_error:
                raise InvalidInputError(f"Value must be at least {min_value}")
            return False
        
        if max_value is not None and value > max_value:
            if raise_on_error:
                raise InvalidInputError(f"Value must be at most {max_value}")
            return False
        
        return True
    
    @staticmethod
    def validate_file_path(
        file_path: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate file path.
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            must_be_file: Whether path must be a file (not directory)
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if not file_path or not isinstance(file_path, str):
            if raise_on_error:
                raise InvalidInputError("File path must be a non-empty string")
            return False
        
        path = Path(file_path)
        
        if must_exist and not path.exists():
            if raise_on_error:
                raise InvalidInputError(f"File does not exist: {file_path}")
            return False
        
        if must_be_file and path.exists() and not path.is_file():
            if raise_on_error:
                raise InvalidInputError(f"Path is not a file: {file_path}")
            return False
        
        return True
    
    @staticmethod
    def validate_audio_format(
        filename: str,
        allowed_formats: Optional[List[str]] = None,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate audio file format.
        
        Args:
            filename: Filename to validate
            allowed_formats: List of allowed extensions (default: common audio formats)
            raise_on_error: Whether to raise exception on error
        
        Returns:
            True if valid
        
        Raises:
            InvalidInputError: If invalid and raise_on_error is True
        """
        if allowed_formats is None:
            allowed_formats = ['wav', 'mp3', 'ogg', 'flac', 'm4a']
        
        if not filename or not isinstance(filename, str):
            if raise_on_error:
                raise InvalidInputError("Filename must be a non-empty string")
            return False
        
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if ext not in allowed_formats:
            if raise_on_error:
                raise InvalidInputError(
                    f"Invalid audio format: {ext}. Allowed: {', '.join(allowed_formats)}"
                )
            return False
        
        return True


# Global validator instance
validator = UnifiedValidator()


# Convenience functions
def validate_uuid(value: str) -> str:
    """Validate UUID and return it."""
    validator.validate_uuid(value, raise_on_error=True)
    return value


def validate_email(value: str) -> str:
    """Validate email and return it."""
    validator.validate_email(value, raise_on_error=True)
    return value


def validate_string(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> str:
    """Validate string and return it."""
    validator.validate_string(value, min_length, max_length, raise_on_error=True)
    return value


def validate_number(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> float:
    """Validate number and return it."""
    validator.validate_number(value, min_value, max_value, raise_on_error=True)
    return float(value)

