"""
Security utilities for Logistics AI Platform

This module provides security-related utilities including input validation,
sanitization, and security checks.
"""

import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from utils.logger import logger
from utils.exceptions import ValidationError, ForbiddenError


class SecurityValidator:
    """Security validation utilities"""
    
    # Common SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bUNION\b.*\bSELECT\b)",
        r"('|(\\')|(;)|(\\;)|(\|)|(\\|))",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"\.\.%2F",
        r"\.\.%5C",
    ]
    
    @classmethod
    def detect_sql_injection(cls, value: str) -> bool:
        """
        Detect potential SQL injection in string
        
        Args:
            value: String to check
            
        Returns:
            True if SQL injection pattern detected
        """
        if not isinstance(value, str):
            return False
        
        value_upper = value.upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {value[:50]}")
                return True
        return False
    
    @classmethod
    def detect_xss(cls, value: str) -> bool:
        """
        Detect potential XSS in string
        
        Args:
            value: String to check
            
        Returns:
            True if XSS pattern detected
        """
        if not isinstance(value, str):
            return False
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {value[:50]}")
                return True
        return False
    
    @classmethod
    def detect_path_traversal(cls, value: str) -> bool:
        """
        Detect potential path traversal in string
        
        Args:
            value: String to check
            
        Returns:
            True if path traversal pattern detected
        """
        if not isinstance(value, str):
            return False
        
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential path traversal detected: {value[:50]}")
                return True
        return False
    
    @classmethod
    def validate_input_security(cls, value: Any, field_name: str = "input") -> None:
        """
        Validate input for security issues
        
        Args:
            value: Value to validate
            field_name: Field name for error messages
            
        Raises:
            ValidationError: If security issue detected
        """
        if isinstance(value, str):
            if cls.detect_sql_injection(value):
                raise ValidationError(
                    f"Invalid input detected in {field_name}",
                    field=field_name
                )
            if cls.detect_xss(value):
                raise ValidationError(
                    f"Invalid input detected in {field_name}",
                    field=field_name
                )
            if cls.detect_path_traversal(value):
                raise ValidationError(
                    f"Invalid input detected in {field_name}",
                    field=field_name
                )
        elif isinstance(value, dict):
            for key, val in value.items():
                cls.validate_input_security(key, f"{field_name}.{key}")
                cls.validate_input_security(val, f"{field_name}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                cls.validate_input_security(item, f"{field_name}[{i}]")


class RateLimitTracker:
    """Track rate limiting per client"""
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize rate limit tracker
        
        Args:
            window_seconds: Time window in seconds
        """
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(
        self,
        client_id: str,
        max_requests: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if client has exceeded rate limit
        
        Args:
            client_id: Client identifier
            max_requests: Maximum requests allowed
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Get requests in current window
        if client_id not in self._requests:
            self._requests[client_id] = []
        
        # Remove old requests
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        request_count = len(self._requests[client_id])
        if request_count >= max_requests:
            return False, 0
        
        # Add current request
        self._requests[client_id].append(now)
        
        remaining = max_requests - request_count - 1
        return True, remaining
    
    def get_remaining(self, client_id: str, max_requests: int) -> int:
        """Get remaining requests for client"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        if client_id not in self._requests:
            return max_requests
        
        request_count = len([
            req_time for req_time in self._requests[client_id]
            if req_time > window_start
        ])
        
        return max(0, max_requests - request_count)
    
    def cleanup_old_entries(self) -> None:
        """Clean up old entries"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [
                req_time for req_time in self._requests[client_id]
                if req_time > window_start
            ]
            
            if not self._requests[client_id]:
                del self._requests[client_id]

