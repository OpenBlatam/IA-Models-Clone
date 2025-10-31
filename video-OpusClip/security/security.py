"""
Security System for Improved Video-OpusClip API

Comprehensive security with:
- Authentication and authorization
- Rate limiting
- Input validation
- Security headers
- Threat detection
- Audit logging
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import hashlib
import hmac
import time
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..config import settings
from ..error_handling import SecurityError, ValidationError

logger = structlog.get_logger("security")

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================

class AuthenticationManager:
    """Manager for authentication operations."""
    
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.JWTError:
            raise SecurityError("Invalid token")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == hashed_password

# =============================================================================
# RATE LIMITING SYSTEM
# =============================================================================

class RateLimiter:
    """Rate limiting system."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.rate_limit_requests = settings.rate_limit_requests
        self.rate_limit_window = settings.rate_limit_window
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for client IP."""
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.rate_limit_window
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True
    
    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client IP."""
        current_time = time.time()
        
        if client_ip not in self.requests:
            return self.rate_limit_requests
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.rate_limit_window
        ]
        
        return max(0, self.rate_limit_requests - len(self.requests[client_ip]))

# =============================================================================
# INPUT VALIDATION SYSTEM
# =============================================================================

class InputValidator:
    """Input validation and sanitization system."""
    
    def __init__(self):
        self.max_url_length = settings.max_url_length
        self.max_title_length = settings.max_title_length
        self.max_description_length = settings.max_description_length
    
    def validate_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL format."""
        if not url or len(url) > self.max_url_length:
            return False
        
        # Basic YouTube URL validation
        youtube_patterns = [
            "youtube.com/watch?v=",
            "youtu.be/",
            "youtube.com/embed/",
            "youtube.com/v/"
        ]
        
        return any(pattern in url.lower() for pattern in youtube_patterns)
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input string."""
        if not input_str:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        return input_str.strip()
    
    def validate_title(self, title: str) -> bool:
        """Validate video title."""
        if not title or len(title) > self.max_title_length:
            return False
        
        # Check for malicious content
        malicious_patterns = [
            'javascript:', 'data:', 'vbscript:', 'onload=', 'onclick='
        ]
        
        title_lower = title.lower()
        return not any(pattern in title_lower for pattern in malicious_patterns)
    
    def validate_description(self, description: str) -> bool:
        """Validate video description."""
        if not description or len(description) > self.max_description_length:
            return False
        
        # Check for malicious content
        malicious_patterns = [
            'javascript:', 'data:', 'vbscript:', '<script', '</script>'
        ]
        
        description_lower = description.lower()
        return not any(pattern in description_lower for pattern in malicious_patterns)

# =============================================================================
# THREAT DETECTION SYSTEM
# =============================================================================

class ThreatDetector:
    """Threat detection and prevention system."""
    
    def __init__(self):
        self.suspicious_patterns = [
            'sql injection', 'xss', 'csrf', 'path traversal',
            'command injection', 'ldap injection', 'xml injection'
        ]
        
        self.malicious_ips = set()  # In production, this would be from a database
        self.blocked_ips = set()
    
    def detect_sql_injection(self, input_str: str) -> bool:
        """Detect SQL injection attempts."""
        sql_patterns = [
            'union select', 'drop table', 'delete from', 'insert into',
            'update set', 'alter table', 'create table', 'exec(',
            'execute(', 'sp_', 'xp_', '--', '/*', '*/'
        ]
        
        input_lower = input_str.lower()
        return any(pattern in input_lower for pattern in sql_patterns)
    
    def detect_xss(self, input_str: str) -> bool:
        """Detect XSS attempts."""
        xss_patterns = [
            '<script', '</script>', 'javascript:', 'onload=',
            'onclick=', 'onerror=', 'onmouseover=', 'onfocus=',
            'onblur=', 'onchange=', 'onsubmit=', 'onreset='
        ]
        
        input_lower = input_str.lower()
        return any(pattern in input_lower for pattern in xss_patterns)
    
    def detect_path_traversal(self, input_str: str) -> bool:
        """Detect path traversal attempts."""
        path_patterns = [
            '../', '..\\', '/etc/passwd', '/etc/shadow',
            'c:\\windows\\system32', 'c:\\boot.ini'
        ]
        
        input_lower = input_str.lower()
        return any(pattern in input_lower for pattern in path_patterns)
    
    def is_suspicious_request(self, request_data: Dict[str, Any]) -> bool:
        """Check if request contains suspicious patterns."""
        for key, value in request_data.items():
            if isinstance(value, str):
                if (self.detect_sql_injection(value) or 
                    self.detect_xss(value) or 
                    self.detect_path_traversal(value)):
                    return True
        return False
    
    def block_ip(self, ip: str):
        """Block IP address."""
        self.blocked_ips.add(ip)
        logger.warning("IP blocked", ip=ip)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips

# =============================================================================
# SECURITY HEADERS SYSTEM
# =============================================================================

class SecurityHeaders:
    """Security headers management."""
    
    def __init__(self):
        self.headers = {
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': settings.content_security_policy
        }
    
    def get_headers(self) -> Dict[str, str]:
        """Get security headers."""
        return self.headers.copy()
    
    def add_custom_header(self, name: str, value: str):
        """Add custom security header."""
        self.headers[name] = value

# =============================================================================
# AUDIT LOGGING SYSTEM
# =============================================================================

class AuditLogger:
    """Audit logging system for security events."""
    
    def __init__(self):
        self.logger = structlog.get_logger("security")
    
    def log_authentication_attempt(self, username: str, success: bool, 
                                 client_ip: str, user_agent: str):
        """Log authentication attempt."""
        self.logger.info(
            "Authentication attempt",
            username=username,
            success=success,
            client_ip=client_ip,
            user_agent=user_agent,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_rate_limit_exceeded(self, client_ip: str, endpoint: str):
        """Log rate limit exceeded."""
        self.logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            endpoint=endpoint,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_security_threat(self, threat_type: str, details: Dict[str, Any], 
                           client_ip: str):
        """Log security threat."""
        self.logger.error(
            "Security threat detected",
            threat_type=threat_type,
            details=details,
            client_ip=client_ip,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_authorization_failure(self, user: str, resource: str, 
                                 client_ip: str):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization failure",
            user=user,
            resource=resource,
            client_ip=client_ip,
            timestamp=datetime.utcnow().isoformat()
        )

# =============================================================================
# SECURITY MANAGER
# =============================================================================

class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        self.threat_detector = ThreatDetector()
        self.security_headers = SecurityHeaders()
        self.audit_logger = AuditLogger()
    
    def validate_request(self, request_data: Dict[str, Any], 
                        client_ip: str) -> bool:
        """Validate request for security threats."""
        
        # Check rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            self.audit_logger.log_rate_limit_exceeded(client_ip, "api")
            raise SecurityError("Rate limit exceeded")
        
        # Check for blocked IP
        if self.threat_detector.is_ip_blocked(client_ip):
            raise SecurityError("IP address is blocked")
        
        # Check for suspicious patterns
        if self.threat_detector.is_suspicious_request(request_data):
            self.audit_logger.log_security_threat(
                "suspicious_request", request_data, client_ip
            )
            raise SecurityError("Suspicious request detected")
        
        return True
    
    def authenticate_user(self, token: str) -> Dict[str, Any]:
        """Authenticate user with JWT token."""
        try:
            payload = self.auth_manager.verify_token(token)
            return payload
        except SecurityError:
            raise
        except Exception as e:
            self.audit_logger.log_authentication_attempt(
                "unknown", False, "unknown", "unknown"
            )
            raise SecurityError("Authentication failed")
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers."""
        return self.security_headers.get_headers()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AuthenticationManager',
    'RateLimiter',
    'InputValidator',
    'ThreatDetector',
    'SecurityHeaders',
    'AuditLogger',
    'SecurityManager'
]






























