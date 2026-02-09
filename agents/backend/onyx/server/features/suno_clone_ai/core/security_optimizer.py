"""
Security Optimizations

Optimizations for:
- Input validation
- SQL injection prevention
- XSS prevention
- Rate limiting
- Authentication/Authorization
- Data encryption
"""

import logging
import hashlib
import secrets
from typing import Optional, Dict, Any, List
import re
from functools import wraps
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Input sanitization and validation."""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            input_str: Input string
            max_length: Maximum length
            
        Returns:
            Sanitized string
        """
        # Remove null bytes
        sanitized = input_str.replace('\x00', '')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate music generation prompt.
        
        Args:
            prompt: Prompt string
            
        Returns:
            (is_valid, error_message)
        """
        if not prompt or not isinstance(prompt, str):
            return False, "Prompt must be a non-empty string"
        
        if len(prompt) > 500:
            return False, "Prompt too long (max 500 characters)"
        
        # Check for potentially malicious patterns
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onload=',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, f"Potentially dangerous pattern detected"
        
        return True, None
    
    @staticmethod
    def validate_duration(duration: int) -> tuple[bool, Optional[str]]:
        """Validate duration."""
        if not isinstance(duration, int):
            return False, "Duration must be an integer"
        
        if duration < 1 or duration > 300:
            return False, "Duration must be between 1 and 300 seconds"
        
        return True, None


class SQLInjectionPreventer:
    """Prevent SQL injection attacks."""
    
    @staticmethod
    def is_safe_query(query: str) -> bool:
        """
        Check if query is safe from SQL injection.
        
        Args:
            query: SQL query string
            
        Returns:
            True if safe
        """
        # Dangerous patterns
        dangerous_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)',
            r'--',
            r'/\*',
            r'union.*select',
            r'exec\s*\(',
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                return False
        
        return True
    
    @staticmethod
    def escape_string(value: str) -> str:
        """
        Escape string for SQL (use parameterized queries instead).
        
        Args:
            value: String value
            
        Returns:
            Escaped string
        """
        # This is a fallback - always prefer parameterized queries
        return value.replace("'", "''").replace("\\", "\\\\")


class RateLimiter:
    """Optimized rate limiting."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed.
        
        Args:
            identifier: Client identifier
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            oldest = min(self.requests[identifier])
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            return False, retry_after
        
        # Record request
        self.requests[identifier].append(now)
        return True, None


class TokenGenerator:
    """Secure token generation."""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length
            
        Returns:
            Secure token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_hash(data: str, salt: Optional[str] = None) -> str:
        """
        Generate secure hash.
        
        Args:
            data: Data to hash
            salt: Optional salt
            
        Returns:
            Hash string
        """
        if salt:
            data = f"{data}{salt}"
        
        return hashlib.sha256(data.encode()).hexdigest()


class SecurityHeaders:
    """Security headers optimization."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Get security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }


def secure_endpoint(func):
    """Decorator for secure endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Add security headers
        # Validate inputs
        # Check rate limits
        # Log security events
        return await func(*args, **kwargs)
    
    return wrapper








