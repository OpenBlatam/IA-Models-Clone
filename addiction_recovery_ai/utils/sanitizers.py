"""
Sanitization utilities
Data sanitization functions
"""

from typing import Any, Optional
import re
import html
from urllib.parse import quote, unquote


def sanitize_html(value: str, escape: bool = True) -> str:
    """
    Sanitize HTML string
    
    Args:
        value: HTML string to sanitize
        escape: Whether to escape HTML entities
    
    Returns:
        Sanitized HTML string
    """
    if not value or not isinstance(value, str):
        return ""
    
    if escape:
        return html.escape(value)
    
    # Remove HTML tags
    return re.sub(r'<[^>]+>', '', value)


def sanitize_filename(value: str) -> str:
    """
    Sanitize filename
    
    Args:
        value: Filename to sanitize
    
    Returns:
        Sanitized filename
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', value)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def sanitize_sql(value: str) -> str:
    """
    Sanitize SQL string (basic)
    
    Args:
        value: SQL string to sanitize
    
    Returns:
        Sanitized SQL string
    
    Note:
        This is basic sanitization. Use parameterized queries for production.
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove SQL injection patterns
    dangerous = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|;|\\)"
    ]
    
    sanitized = value
    for pattern in dangerous:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized


def sanitize_url(value: str) -> str:
    """
    Sanitize URL
    
    Args:
        value: URL to sanitize
    
    Returns:
        Sanitized URL
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
    value_lower = value.lower()
    
    for protocol in dangerous_protocols:
        if value_lower.startswith(protocol):
            return ""
    
    # URL encode
    return quote(value, safe='/:?=&')


def sanitize_email(value: str) -> str:
    """
    Sanitize email address
    
    Args:
        value: Email to sanitize
    
    Returns:
        Sanitized email
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove whitespace
    sanitized = value.strip().lower()
    
    # Remove invalid characters
    sanitized = re.sub(r'[^\w@._-]', '', sanitized)
    
    return sanitized


def sanitize_phone(value: str) -> str:
    """
    Sanitize phone number
    
    Args:
        value: Phone number to sanitize
    
    Returns:
        Sanitized phone number
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove all non-digit characters except +
    sanitized = re.sub(r'[^\d+]', '', value)
    
    # Ensure + is only at the start
    if '+' in sanitized and not sanitized.startswith('+'):
        sanitized = sanitized.replace('+', '')
    
    return sanitized


def sanitize_string(
    value: str,
    max_length: Optional[int] = None,
    remove_whitespace: bool = False,
    remove_special: bool = False
) -> str:
    """
    Sanitize string
    
    Args:
        value: String to sanitize
        max_length: Maximum length
        remove_whitespace: Remove whitespace
        remove_special: Remove special characters
    
    Returns:
        Sanitized string
    """
    if not value or not isinstance(value, str):
        return ""
    
    sanitized = value
    
    if remove_whitespace:
        sanitized = re.sub(r'\s+', '', sanitized)
    
    if remove_special:
        sanitized = re.sub(r'[^\w\s]', '', sanitized)
    
    if max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def sanitize_number(value: Any, default: float = 0.0) -> float:
    """
    Sanitize number
    
    Args:
        value: Value to sanitize
        default: Default value if invalid
    
    Returns:
        Sanitized number
    """
    try:
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d.-]', '', value)
            return float(cleaned)
        
        return default
    except (ValueError, TypeError):
        return default


def sanitize_boolean(value: Any, default: bool = False) -> bool:
    """
    Sanitize boolean
    
    Args:
        value: Value to sanitize
        default: Default value if invalid
    
    Returns:
        Sanitized boolean
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ('true', '1', 'yes', 'on'):
            return True
        if lower in ('false', '0', 'no', 'off'):
            return False
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return default


def remove_whitespace(value: str) -> str:
    """
    Remove all whitespace
    
    Args:
        value: String to process
    
    Returns:
        String without whitespace
    """
    if not value or not isinstance(value, str):
        return ""
    
    return re.sub(r'\s+', '', value)


def normalize_whitespace(value: str) -> str:
    """
    Normalize whitespace (multiple spaces to single space)
    
    Args:
        value: String to process
    
    Returns:
        String with normalized whitespace
    """
    if not value or not isinstance(value, str):
        return ""
    
    return re.sub(r'\s+', ' ', value).strip()


def remove_special_chars(value: str, keep: str = "") -> str:
    """
    Remove special characters
    
    Args:
        value: String to process
        keep: Characters to keep
    
    Returns:
        String without special characters
    """
    if not value or not isinstance(value, str):
        return ""
    
    pattern = f'[^\\w\\s{re.escape(keep)}]'
    return re.sub(pattern, '', value)

