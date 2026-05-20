"""
Validation utilities
Data validation functions
"""

from typing import Any, Optional, Callable, List
import re
from email.utils import parseaddr


def is_email(value: str) -> bool:
    """
    Validate email address
    
    Args:
        value: Email address to validate
    
    Returns:
        True if valid email
    """
    if not value or not isinstance(value, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))


def is_url(value: str) -> bool:
    """
    Validate URL
    
    Args:
        value: URL to validate
    
    Returns:
        True if valid URL
    """
    if not value or not isinstance(value, str):
        return False
    
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?$'
    return bool(re.match(pattern, value))


def is_phone(value: str) -> bool:
    """
    Validate phone number
    
    Args:
        value: Phone number to validate
    
    Returns:
        True if valid phone number
    """
    if not value or not isinstance(value, str):
        return False
    
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)]', '', value)
    
    # Check if all digits (with optional + prefix)
    if cleaned.startswith('+'):
        cleaned = cleaned[1:]
    
    return cleaned.isdigit() and len(cleaned) >= 10


def is_uuid(value: str) -> bool:
    """
    Validate UUID
    
    Args:
        value: UUID to validate
    
    Returns:
        True if valid UUID
    """
    if not value or not isinstance(value, str):
        return False
    
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(pattern, value, re.IGNORECASE))


def is_strong_password(value: str, min_length: int = 8) -> bool:
    """
    Validate strong password
    
    Args:
        value: Password to validate
        min_length: Minimum length
    
    Returns:
        True if strong password
    """
    if not value or not isinstance(value, str):
        return False
    
    if len(value) < min_length:
        return False
    
    has_upper = bool(re.search(r'[A-Z]', value))
    has_lower = bool(re.search(r'[a-z]', value))
    has_digit = bool(re.search(r'\d', value))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', value))
    
    return has_upper and has_lower and has_digit and has_special


def is_credit_card(value: str) -> bool:
    """
    Validate credit card number (Luhn algorithm)
    
    Args:
        value: Credit card number to validate
    
    Returns:
        True if valid credit card number
    """
    if not value or not isinstance(value, str):
        return False
    
    # Remove spaces and dashes
    cleaned = re.sub(r'[\s\-]', '', value)
    
    if not cleaned.isdigit():
        return False
    
    # Luhn algorithm
    def luhn_check(card_num: str) -> bool:
        def digits_of(n: str) -> List[int]:
            return [int(d) for d in n]
        
        digits = digits_of(card_num)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        
        for d in even_digits:
            checksum += sum(digits_of(str(d * 2)))
        
        return checksum % 10 == 0
    
    return luhn_check(cleaned)


def is_ip_address(value: str, version: int = 4) -> bool:
    """
    Validate IP address
    
    Args:
        value: IP address to validate
        version: IP version (4 or 6)
    
    Returns:
        True if valid IP address
    """
    if not value or not isinstance(value, str):
        return False
    
    if version == 4:
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, value):
            return False
        
        parts = value.split('.')
        return all(0 <= int(part) <= 255 for part in parts)
    
    if version == 6:
        pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        return bool(re.match(pattern, value))
    
    return False


def is_alpha(value: str) -> bool:
    """
    Validate alphabetic string
    
    Args:
        value: String to validate
    
    Returns:
        True if only alphabetic characters
    """
    if not value or not isinstance(value, str):
        return False
    
    return value.isalpha()


def is_alphanumeric(value: str) -> bool:
    """
    Validate alphanumeric string
    
    Args:
        value: String to validate
    
    Returns:
        True if only alphanumeric characters
    """
    if not value or not isinstance(value, str):
        return False
    
    return value.isalnum()


def is_numeric(value: str) -> bool:
    """
    Validate numeric string
    
    Args:
        value: String to validate
    
    Returns:
        True if only numeric characters
    """
    if not value or not isinstance(value, str):
        return False
    
    return value.isdigit()


def is_in_range(value: float, min_val: float, max_val: float) -> bool:
    """
    Validate value is in range
    
    Args:
        value: Value to validate
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        True if value is in range
    """
    return min_val <= value <= max_val


def is_length(value: str, min_len: Optional[int] = None, max_len: Optional[int] = None) -> bool:
    """
    Validate string length
    
    Args:
        value: String to validate
        min_len: Minimum length
        max_len: Maximum length
    
    Returns:
        True if length is valid
    """
    if not isinstance(value, str):
        return False
    
    length = len(value)
    
    if min_len is not None and length < min_len:
        return False
    
    if max_len is not None and length > max_len:
        return False
    
    return True


def validate_with(
    value: Any,
    validator: Callable[[Any], bool],
    error_message: str = "Validation failed"
) -> tuple[bool, Optional[str]]:
    """
    Validate value with custom validator
    
    Args:
        value: Value to validate
        validator: Validator function
        error_message: Error message if validation fails
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if validator(value):
            return True, None
        return False, error_message
    except Exception as e:
        return False, f"Validation error: {str(e)}"
