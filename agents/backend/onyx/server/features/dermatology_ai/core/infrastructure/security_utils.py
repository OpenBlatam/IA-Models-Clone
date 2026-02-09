"""
Security Utilities
Input sanitization and security helpers
"""

import re
import html
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Utility class for sanitizing user inputs"""
    
    # Patterns for potentially dangerous content
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize string input
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Trim whitespace
        value = value.strip()
        
        # Check length
        if max_length and len(value) > max_length:
            logger.warning(f"Input truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]
        
        return value
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitize HTML input by escaping special characters
        
        Args:
            value: HTML string to sanitize
            
        Returns:
            Escaped HTML string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Escape HTML entities
        return html.escape(value, quote=True)
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """
        Check if string contains potential SQL injection patterns
        
        Args:
            value: String to check
            
        Returns:
            True if suspicious patterns found
        """
        if not isinstance(value, str):
            return False
        
        value_upper = value.upper()
        for pattern in InputSanitizer.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {pattern}")
                return True
        
        return False
    
    @staticmethod
    def check_xss(value: str) -> bool:
        """
        Check if string contains potential XSS patterns
        
        Args:
            value: String to check
            
        Returns:
            True if suspicious patterns found
        """
        if not isinstance(value, str):
            return False
        
        for pattern in InputSanitizer.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {pattern}")
                return True
        
        return False
    
    @staticmethod
    def sanitize_user_input(value: Any, input_type: str = "string") -> Any:
        """
        Comprehensive input sanitization
        
        Args:
            value: Input value to sanitize
            input_type: Type of input (string, html, etc.)
            
        Returns:
            Sanitized value
        """
        if value is None:
            return None
        
        if input_type == "html":
            if isinstance(value, str):
                # Check for XSS
                if InputSanitizer.check_xss(value):
                    raise ValueError("Potentially dangerous content detected")
                return InputSanitizer.sanitize_html(value)
        
        if isinstance(value, str):
            # Check for SQL injection
            if InputSanitizer.check_sql_injection(value):
                raise ValueError("Potentially dangerous content detected")
            
            # Basic sanitization
            return InputSanitizer.sanitize_string(value)
        
        return value


class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_id_format(id_value: str, max_length: int = 255) -> bool:
        """
        Validate ID format (UUID, alphanumeric, etc.)
        
        Args:
            id_value: ID to validate
            max_length: Maximum length
            
        Returns:
            True if valid format
        """
        if not isinstance(id_value, str):
            return False
        
        if len(id_value) > max_length:
            return False
        
        # Allow alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', id_value):
            return False
        
        return True
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Basic email validation
        
        Args:
            email: Email to validate
            
        Returns:
            True if valid email format
        """
        if not isinstance(email, str):
            return False
        
        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Basic URL validation
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid URL format
        """
        if not isinstance(url, str):
            return False
        
        # Basic URL pattern
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))















