"""
Security utilities for input sanitization and validation.
"""

import re
import html
from typing import Any, Dict, List


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks.
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    return html.escape(text)


def sanitize_sql_input(text: str) -> str:
    """
    Basic SQL injection prevention (for demonstration).
    In production, use parameterized queries.
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    dangerous_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\*|'|"|\\|/)",
    ]
    
    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
    return sanitized


def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format (basic check).
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if format is valid
    """
    if not api_key or len(api_key) < 20:
        return False
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
        return False
    
    return True


def rate_limit_key(identifier: str, endpoint: str) -> str:
    """
    Generate a rate limit key.
    
    Args:
        identifier: Client identifier
        endpoint: API endpoint
    
    Returns:
        Rate limit key
    """
    return f"rate_limit:{identifier}:{endpoint}"


def generate_csrf_token() -> str:
    """
    Generate a simple CSRF token (for demonstration).
    In production, use a proper CSRF token library.
    
    Returns:
        CSRF token
    """
    import secrets
    return secrets.token_urlsafe(32)






