"""
Security Manager for Color Grading AI
======================================

Advanced security features and threat protection.
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Advanced security manager.
    
    Features:
    - Input validation and sanitization
    - Path traversal protection
    - SQL injection prevention
    - XSS protection
    - Rate limiting per user
    - Threat detection
    """
    
    def __init__(self):
        """Initialize security manager."""
        self._suspicious_activities: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked_ips: set = set()
        self._rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self._max_requests_per_minute = 60
        self._max_suspicious_activities = 5
    
    def validate_input(self, input_data: Any, input_type: str = "general") -> bool:
        """
        Validate and sanitize input.
        
        Args:
            input_data: Input data to validate
            input_type: Type of input (path, url, text, etc.)
            
        Returns:
            True if valid
        """
        if input_type == "path":
            return self._validate_path(input_data)
        elif input_type == "url":
            return self._validate_url(input_data)
        elif input_type == "text":
            return self._validate_text(input_data)
        return True
    
    def _validate_path(self, path: str) -> bool:
        """Validate file path (prevent path traversal)."""
        # Check for path traversal attempts
        dangerous_patterns = [
            "..",
            "//",
            "\\",
            "~",
            "/etc/",
            "/proc/",
            "C:\\",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in path:
                logger.warning(f"Path traversal attempt detected: {path}")
                return False
        
        return True
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        return bool(url_pattern.match(url))
    
    def _validate_text(self, text: str) -> bool:
        """Validate text (prevent XSS, SQL injection)."""
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected")
                return False
        
        # Check for XSS patterns
        xss_patterns = [
            r"<script",
            r"javascript:",
            r"onerror=",
            r"onload=",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"XSS attempt detected")
                return False
        
        return True
    
    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check rate limit for identifier.
        
        Args:
            identifier: User/IP identifier
            
        Returns:
            True if within limit
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Clean old requests
        self._rate_limits[identifier] = [
            req_time for req_time in self._rate_limits[identifier]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self._rate_limits[identifier]) >= self._max_requests_per_minute:
            self._record_suspicious_activity(identifier, "rate_limit_exceeded")
            return False
        
        # Add current request
        self._rate_limits[identifier].append(now)
        return True
    
    def _record_suspicious_activity(self, identifier: str, activity_type: str):
        """Record suspicious activity."""
        self._suspicious_activities[identifier].append(datetime.now())
        
        # Clean old activities
        cutoff = datetime.now() - timedelta(hours=1)
        self._suspicious_activities[identifier] = [
            activity for activity in self._suspicious_activities[identifier]
            if activity > cutoff
        ]
        
        # Block if too many suspicious activities
        if len(self._suspicious_activities[identifier]) >= self._max_suspicious_activities:
            self._blocked_ips.add(identifier)
            logger.warning(f"Blocked {identifier} due to suspicious activities")
    
    def is_blocked(self, identifier: str) -> bool:
        """Check if identifier is blocked."""
        return identifier in self._blocked_ips
    
    def unblock(self, identifier: str):
        """Unblock identifier."""
        self._blocked_ips.discard(identifier)
        if identifier in self._suspicious_activities:
            del self._suspicious_activities[identifier]
        logger.info(f"Unblocked {identifier}")
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename."""
        # Remove dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        return sanitized
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """Hash password with salt."""
        if not salt:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode())
        return f"{salt}:{hash_obj.hexdigest()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_value = hashed.split(":")
            computed_hash = self.hash_password(password, salt)
            return computed_hash == hashed
        except:
            return False
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "blocked_ips": len(self._blocked_ips),
            "suspicious_activities": sum(len(activities) for activities in self._suspicious_activities.values()),
            "rate_limited_users": len([u for u, reqs in self._rate_limits.items() if len(reqs) >= self._max_requests_per_minute]),
        }




