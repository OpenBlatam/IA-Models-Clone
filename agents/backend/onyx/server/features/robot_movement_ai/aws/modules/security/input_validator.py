"""
Input Validator
===============

Advanced input validation and sanitization.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator, ValidationError

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validator with sanitization."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"\.\.%2F",
        r"\.\.%5C",
    ]
    
    @classmethod
    def validate_string(
        cls,
        value: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_html: bool = False
    ) -> str:
        """Validate and sanitize string."""
        if not isinstance(value, str):
            raise ValueError("Value must be a string")
        
        # Length validation
        if max_length and len(value) > max_length:
            raise ValueError(f"String exceeds maximum length of {max_length}")
        if min_length and len(value) < min_length:
            raise ValueError(f"String below minimum length of {min_length}")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            raise ValueError(f"String does not match required pattern")
        
        # Sanitization
        sanitized = cls.sanitize(value, allow_html=allow_html)
        
        return sanitized
    
    @classmethod
    def sanitize(cls, value: str, allow_html: bool = False) -> str:
        """Sanitize string input."""
        # Remove SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        # Remove XSS patterns
        if not allow_html:
            for pattern in cls.XSS_PATTERNS:
                value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        # Remove path traversal patterns
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        return value.strip()
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email address."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return cls.sanitize(email)
    
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """Validate URL."""
        allowed_schemes = allowed_schemes or ["http", "https"]
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            if parsed.scheme not in allowed_schemes:
                raise ValueError(f"URL scheme must be one of: {allowed_schemes}")
            
            if not parsed.netloc:
                raise ValueError("URL must have a valid domain")
            
            return url
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}")
    
    @classmethod
    def validate_number(
        cls,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """Validate number."""
        try:
            num = float(value)
        except (ValueError, TypeError):
            raise ValueError("Value must be a number")
        
        if min_value is not None and num < min_value:
            raise ValueError(f"Number must be at least {min_value}")
        if max_value is not None and num > max_value:
            raise ValueError(f"Number must be at most {max_value}")
        
        return num















