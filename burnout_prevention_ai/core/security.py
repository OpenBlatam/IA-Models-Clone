"""
Security Utilities
==================
Security-related utilities and helpers.

Provides input sanitization, validation, and security headers
for protecting the application from malicious input and attacks.
"""

import re
from typing import Optional


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input by removing potentially dangerous characters.
    
    Args:
        value: String to sanitize (will be converted to string if not already)
        max_length: Optional maximum length
        
    Returns:
        Sanitized string
    """
    if value is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)
    
    if not value:
        return ""
    
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', value)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Apply length limit if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def sanitize_list(items: list, max_items: Optional[int] = None) -> list:
    """
    Sanitize list of strings.
    
    Args:
        items: List of items to sanitize (validated to be a list)
        max_items: Optional maximum number of items
        
    Returns:
        Sanitized list
        
    Raises:
        TypeError: If items is not a list
    """
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    
    if not items:
        return []
    
    # Filter out None/empty values and sanitize
    sanitized = [sanitize_string(str(item)) for item in items if item is not None]
    
    # Remove empty strings after sanitization
    sanitized = [item for item in sanitized if item]
    
    if max_items and len(sanitized) > max_items:
        sanitized = sanitized[:max_items]
    
    return sanitized


def validate_numeric_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error message
        
    Raises:
        ValueError if value is out of range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")


def get_security_headers() -> dict:
    """
    Get security headers for responses.
    
    Returns comprehensive security headers to protect against common attacks:
    - XSS (Cross-Site Scripting)
    - Clickjacking
    - MIME type sniffing
    - Referrer leakage
    
    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"  # HSTS for HTTPS
    }

