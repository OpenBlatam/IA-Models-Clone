"""
Security utilities for input validation and sanitization.
"""

from typing import Any, Optional
import re
import html
import logging

logger = logging.getLogger(__name__)


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML by escaping special characters.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    return html.escape(text)


def sanitize_sql_input(value: Any) -> str:
    """
    Sanitize input to prevent SQL injection.
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized string
    """
    if value is None:
        return ""
    
    # Convert to string and remove dangerous characters
    text = str(value)
    # Remove SQL injection patterns
    text = re.sub(r'[;\'"\\]', '', text)
    
    return text.strip()


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return ""
    
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.strip('. ')
    
    return filename[:255]  # Limit length


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Token length
        
    Returns:
        Secure random token
    """
    import secrets
    import string
    
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data (simple implementation).
    
    Args:
        data: Data to hash
        
    Returns:
        Hashed data
    """
    import hashlib
    
    return hashlib.sha256(data.encode()).hexdigest()


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input to prevent XSS and injection attacks.
    
    Args:
        text: Text to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip whitespace
    text = text.strip()
    
    # Apply length limit if specified
    if max_length:
        text = text[:max_length]
    
    return text


def hash_data(data: str) -> str:
    """
    Hash data using SHA-256.
    
    Args:
        data: Data to hash
        
    Returns:
        Hashed data (hex digest)
    """
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()





