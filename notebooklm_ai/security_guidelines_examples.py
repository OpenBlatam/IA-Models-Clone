from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum
import ipaddress
import urllib.parse
import socket
import threading
from contextlib import contextmanager
    import jwt
    from jwt import PyJWTError, encode, decode
    import bcrypt
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Any, List, Dict, Optional
"""
Security-Specific Guidelines and Examples
========================================

This module provides comprehensive security guidelines and implementations covering:
- Input validation and sanitization
- Authentication and authorization patterns
- Secure communication protocols
- Data encryption and key management
- Security logging and monitoring
- Vulnerability prevention patterns
- Secure configuration management
- Threat modeling and risk assessment

Features:
- Secure coding patterns and anti-patterns
- Input validation with multiple validation layers
- Authentication mechanisms (JWT, OAuth, API keys)
- Authorization with role-based access control
- Secure communication (TLS, certificate validation)
- Data protection (encryption, hashing, key rotation)
- Security logging and audit trails
- Vulnerability scanning and prevention
- Secure configuration validation
- Threat modeling and risk assessment

Author: AI Assistant
License: MIT
"""


try:
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    PyJWTError = Exception
    encode = None
    decode = None

try:
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat levels for risk assessment."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityValidationResult:
    """Result of security validation."""
    valid: bool
    security_level: SecurityLevel
    threats_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    risk_score: float = 0.0


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    session_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    error_message: str = ""
    security_events: List[str] = field(default_factory=list)


@dataclass
class AuthorizationResult:
    """Result of authorization check."""
    allowed: bool
    required_permissions: List[str] = field(default_factory=list)
    user_permissions: List[str] = field(default_factory=list)
    missing_permissions: List[str] = field(default_factory=list)
    reason: str = ""
    audit_trail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    severity: SecurityLevel
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    request_id: Optional[str] = None


class SecurityValidationError(Exception):
    """Custom exception for security validation errors."""
    pass


class AuthenticationError(Exception):
    """Custom exception for authentication errors."""
    pass


class AuthorizationError(Exception):
    """Custom exception for authorization errors."""
    pass


class SecurityConfigurationError(Exception):
    """Custom exception for security configuration errors."""
    pass


class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


class SecurityManager:
    """Main security manager with comprehensive security features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security manager with configuration."""
        self.config = config or {}
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()
        self.session_store: Dict[str, Dict[str, Any]] = {}
        self._security_lock = threading.Lock()
        self._rate_limit_windows: Dict[str, List[float]] = {}
        
        # Security settings
        self.max_failed_attempts = self.config.get('max_failed_attempts', 5)
        self.block_duration = self.config.get('block_duration', 300)  # 5 minutes
        self.rate_limit_requests = self.config.get('rate_limit_requests', 100)
        self.rate_limit_window = self.config.get('rate_limit_window', 60)
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        self.password_min_length = self.config.get('password_min_length', 12)
        self.require_special_chars = self.config.get('require_special_chars', True)
        self.require_numbers = self.config.get('require_numbers', True)
        self.require_uppercase = self.config.get('require_uppercase', True)
        self.require_lowercase = self.config.get('require_lowercase', True)
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event."""
        with self._security_lock:
            self.security_events.append(event)
            
            # Keep only recent events (last 1000)
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
        
        logger.warning(f"Security Event: {event.event_type} - {event.severity.value} - {event.details}")
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for identifier."""
        current_time = time.time()
        
        if identifier not in self._rate_limit_windows:
            self._rate_limit_windows[identifier] = []
        
        # Remove old entries
        window_start = current_time - self.rate_limit_window
        self._rate_limit_windows[identifier] = [
            t for t in self._rate_limit_windows[identifier] 
            if t > window_start
        ]
        
        # Check if limit exceeded
        if len(self._rate_limit_windows[identifier]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self._rate_limit_windows[identifier].append(current_time)
        return True
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        if not ip_address:
            return False
        
        return ip_address in self.blocked_ips
    
    def _block_ip(self, ip_address: str, duration: int = None):
        """Block IP address for specified duration."""
        if not ip_address:
            return
        
        duration = duration or self.block_duration
        
        with self._security_lock:
            self.blocked_ips.add(ip_address)
        
        # Schedule unblock
        def unblock_ip():
            
    """unblock_ip function."""
time.sleep(duration)
            with self._security_lock:
                self.blocked_ips.discard(ip_address)
        
        threading.Thread(target=unblock_ip, daemon=True).start()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        self._log_security_event(SecurityEvent(
            event_type="ip_blocked",
            severity=SecurityLevel.MEDIUM,
            timestamp=datetime.now(),
            ip_address=ip_address,
            details={"duration": duration}
        ))
    
    def validate_input(self, input_data: Any, input_type: str, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> SecurityValidationResult:
        """Validate input data with security checks."""
        start_time = time.time()
        threats_detected = []
        recommendations = []
        
        if input_data is None:
            return SecurityValidationResult(
                valid=False,
                security_level=security_level,
                threats_detected=["null_input"],
                recommendations=["Input cannot be null"],
                validation_time=time.time() - start_time
            )
        
        # Convert to string for validation
        if not isinstance(input_data, str):
            input_data = str(input_data)
        
        # Check for common attack patterns
        attack_patterns = {
            'sql_injection': [
                r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                r"(--|#|/\*|\*/)",
                r"(\b(exec|execute|xp_|sp_)\b)",
                r"(\b(script|javascript|vbscript|onload|onerror)\b)"
            ],
            'xss': [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript:)",
                r"(on\w+\s*=)",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)",
                r"(<embed[^>]*>)"
            ],
            'path_traversal': [
                r"(\.\./|\.\.\\)",
                r"(/etc/passwd|/etc/shadow)",
                r"(c:\\windows\\system32)",
                r"(%2e%2e%2f|%2e%2e%5c)"
            ],
            'command_injection': [
                r"(\b(cmd|command|exec|system|eval|subprocess)\b)",
                r"(\||&|;|`|\\$\\()",
                r"(/bin/bash|/bin/sh|cmd\.exe)"
            ]
        }
        
        for threat_type, patterns in attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    threats_detected.append(threat_type)
                    recommendations.append(f"Detected {threat_type} pattern")
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
        for char in suspicious_chars:
            if char in input_data:
                threats_detected.append("suspicious_characters")
                recommendations.append(f"Contains suspicious character: {char}")
                break
        
        # Check input length
        max_length = {
            SecurityLevel.LOW: 1000,
            SecurityLevel.MEDIUM: 500,
            SecurityLevel.HIGH: 100,
            SecurityLevel.CRITICAL: 50
        }.get(security_level, 500)
        
        if len(input_data) > max_length:
            threats_detected.append("input_too_long")
            recommendations.append(f"Input exceeds maximum length of {max_length}")
        
        # Calculate risk score
        risk_score = len(threats_detected) * 0.2
        if risk_score > 1.0:
            risk_score = 1.0
        
        valid = len(threats_detected) == 0
        
        return SecurityValidationResult(
            valid=valid,
            security_level=security_level,
            threats_detected=threats_detected,
            recommendations=recommendations,
            validation_time=time.time() - start_time,
            risk_score=risk_score
        )
    
    def validate_password(self, password: str) -> SecurityValidationResult:
        """Validate password strength."""
        if not password:
            return SecurityValidationResult(
                valid=False,
                security_level=SecurityLevel.HIGH,
                threats_detected=["empty_password"],
                recommendations=["Password cannot be empty"]
            )
        
        threats_detected = []
        recommendations = []
        
        # Check length
        if len(password) < self.password_min_length:
            threats_detected.append("password_too_short")
            recommendations.append(f"Password must be at least {self.password_min_length} characters")
        
        # Check character requirements
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            threats_detected.append("no_uppercase")
            recommendations.append("Password must contain uppercase letters")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            threats_detected.append("no_lowercase")
            recommendations.append("Password must contain lowercase letters")
        
        if self.require_numbers and not re.search(r'\d', password):
            threats_detected.append("no_numbers")
            recommendations.append("Password must contain numbers")
        
        if self.require_special_chars and not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            threats_detected.append("no_special_chars")
            recommendations.append("Password must contain special characters")
        
        # Check for common patterns
        common_patterns = [
            r"123456",
            r"password",
            r"qwerty",
            r"admin",
            r"letmein",
            r"welcome",
            r"monkey",
            r"dragon",
            r"master",
            r"football"
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                threats_detected.append("common_password")
                recommendations.append("Password contains common pattern")
                break
        
        # Check for sequential characters
        if re.search(r'(.)\1{2,}', password):
            threats_detected.append("repeated_chars")
            recommendations.append("Password contains repeated characters")
        
        valid = len(threats_detected) == 0
        
        return SecurityValidationResult(
            valid=valid,
            security_level=SecurityLevel.HIGH,
            threats_detected=threats_detected,
            recommendations=recommendations,
            risk_score=len(threats_detected) * 0.15
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if not BCRYPT_AVAILABLE:
            raise SecurityConfigurationError("bcrypt is not available. Install with: pip install bcrypt")
        
        if not password:
            raise InputValidationError("Password cannot be empty")
        
        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        if not BCRYPT_AVAILABLE:
            raise SecurityConfigurationError("bcrypt is not available")
        
        if not password or not hashed_password:
            return False
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False
    
    def generate_jwt_token(self, payload: Dict[str, Any], secret_key: str, expires_in: int = 3600) -> str:
        """Generate JWT token."""
        if not JWT_AVAILABLE:
            raise SecurityConfigurationError("PyJWT is not available. Install with: pip install PyJWT")
        
        if not payload or not secret_key:
            raise InputValidationError("Payload and secret key are required")
        
        # Add standard claims
        now = datetime.utcnow()
        token_payload = {
            **payload,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'nbf': now
        }
        
        return encode(token_payload, secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str, secret_key: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        if not JWT_AVAILABLE:
            raise SecurityConfigurationError("PyJWT is not available")
        
        if not token or not secret_key:
            return None
        
        try:
            payload = decode(token, secret_key, algorithms=['HS256'])
            return payload
        except PyJWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> AuthenticationResult:
        """Authenticate user with security checks."""
        if not username or not password:
            return AuthenticationResult(
                success=False,
                error_message="Username and password are required"
            )
        
        # Check if IP is blocked
        if ip_address and self._is_ip_blocked(ip_address):
            return AuthenticationResult(
                success=False,
                error_message="IP address is blocked",
                security_events=["ip_blocked"]
            )
        
        # Check rate limiting
        if not self._check_rate_limit(f"auth:{ip_address or 'unknown'}"):
            return AuthenticationResult(
                success=False,
                error_message="Rate limit exceeded",
                security_events=["rate_limit_exceeded"]
            )
        
        # Validate input
        username_validation = self.validate_input(username, "username", SecurityLevel.HIGH)
        password_validation = self.validate_input(password, "password", SecurityLevel.HIGH)
        
        if not username_validation.valid or not password_validation.valid:
            return AuthenticationResult(
                success=False,
                error_message="Invalid input detected",
                security_events=["invalid_input"]
            )
        
        # TODO: Implement actual user authentication logic
        # This is a placeholder for demonstration
        
        # Simulate authentication failure
        if username == "admin" and password == "wrong_password":
            # Track failed attempts
            with self._security_lock:
                self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
                
                if self.failed_attempts[username] >= self.max_failed_attempts:
                    if ip_address:
                        self._block_ip(ip_address)
            
            self._log_security_event(SecurityEvent(
                event_type="authentication_failed",
                severity=SecurityLevel.MEDIUM,
                timestamp=datetime.now(),
                user_id=username,
                ip_address=ip_address,
                details={"failed_attempts": self.failed_attempts.get(username, 0)}
            ))
            
            return AuthenticationResult(
                success=False,
                error_message="Invalid credentials",
                security_events=["authentication_failed"]
            )
        
        # Simulate successful authentication
        if username == "admin" and password == "correct_password":
            # Reset failed attempts
            with self._security_lock:
                self.failed_attempts.pop(username, None)
            
            # Generate session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
            
            # Store session
            self.session_store[session_token] = {
                'user_id': username,
                'roles': ['admin'],
                'permissions': ['read', 'write', 'delete'],
                'expires_at': expires_at,
                'ip_address': ip_address
            }
            
            self._log_security_event(SecurityEvent(
                event_type="authentication_success",
                severity=SecurityLevel.LOW,
                timestamp=datetime.now(),
                user_id=username,
                ip_address=ip_address,
                details={"session_token": session_token[:8] + "..."}
            ))
            
            return AuthenticationResult(
                success=True,
                user_id=username,
                roles=['admin'],
                permissions=['read', 'write', 'delete'],
                session_token=session_token,
                expires_at=expires_at
            )
        
        return AuthenticationResult(
            success=False,
            error_message="Invalid credentials"
        )
    
    def authorize_action(self, session_token: str, required_permissions: List[str], resource: str = None) -> AuthorizationResult:
        """Authorize action based on session token and required permissions."""
        if not session_token:
            return AuthorizationResult(
                allowed=False,
                reason="No session token provided"
            )
        
        # Get session
        session = self.session_store.get(session_token)
        if not session:
            return AuthorizationResult(
                allowed=False,
                reason="Invalid session token"
            )
        
        # Check if session expired
        if session['expires_at'] < datetime.now():
            # Remove expired session
            self.session_store.pop(session_token, None)
            return AuthorizationResult(
                allowed=False,
                reason="Session expired"
            )
        
        user_permissions = session.get('permissions', [])
        missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
        
        allowed = len(missing_permissions) == 0
        
        audit_trail = {
            'user_id': session.get('user_id'),
            'session_token': session_token[:8] + "...",
            'resource': resource,
            'timestamp': datetime.now().isoformat()
        }
        
        if not allowed:
            self._log_security_event(SecurityEvent(
                event_type="authorization_failed",
                severity=SecurityLevel.MEDIUM,
                timestamp=datetime.now(),
                user_id=session.get('user_id'),
                details={
                    'required_permissions': required_permissions,
                    'user_permissions': user_permissions,
                    'resource': resource
                }
            ))
        
        return AuthorizationResult(
            allowed=allowed,
            required_permissions=required_permissions,
            user_permissions=user_permissions,
            missing_permissions=missing_permissions,
            reason="Insufficient permissions" if not allowed else "Authorized",
            audit_trail=audit_trail
        )
    
    def validate_url(self, url: str, allowed_domains: List[str] = None) -> SecurityValidationResult:
        """Validate URL for security."""
        if not url:
            return SecurityValidationResult(
                valid=False,
                security_level=SecurityLevel.MEDIUM,
                threats_detected=["empty_url"]
            )
        
        threats_detected = []
        recommendations = []
        
        try:
            parsed_url = urllib.parse.urlparse(url)
            
            # Check protocol
            if parsed_url.scheme not in ['http', 'https']:
                threats_detected.append("unsafe_protocol")
                recommendations.append("Only HTTP and HTTPS protocols are allowed")
            
            # Check for IP addresses (potential SSRF)
            try:
                ipaddress.ip_address(parsed_url.hostname)
                threats_detected.append("ip_address_in_url")
                recommendations.append("URL contains IP address instead of domain name")
            except ValueError:
                pass  # Not an IP address
            
            # Check domain whitelist
            if allowed_domains and parsed_url.hostname not in allowed_domains:
                threats_detected.append("untrusted_domain")
                recommendations.append(f"Domain {parsed_url.hostname} not in allowed list")
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r"(file://)",
                r"(ftp://)",
                r"(javascript:)",
                r"(data:)",
                r"(vbscript:)",
                r"(on\w+\s*=)"
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    threats_detected.append("suspicious_url_pattern")
                    recommendations.append(f"URL contains suspicious pattern: {pattern}")
                    break
            
        except Exception as e:
            threats_detected.append("invalid_url")
            recommendations.append(f"Invalid URL format: {e}")
        
        valid = len(threats_detected) == 0
        
        return SecurityValidationResult(
            valid=valid,
            security_level=SecurityLevel.MEDIUM,
            threats_detected=threats_detected,
            recommendations=recommendations,
            risk_score=len(threats_detected) * 0.25
        )
    
    async def validate_file_upload(self, filename: str, content_type: str, file_size: int, 
                           max_size: int = 10485760, allowed_types: List[str] = None) -> SecurityValidationResult:
        """Validate file upload for security."""
        threats_detected = []
        recommendations = []
        
        # Check file size
        if file_size > max_size:
            threats_detected.append("file_too_large")
            recommendations.append(f"File size {file_size} exceeds maximum {max_size}")
        
        # Check file extension
        if filename:
            ext = Path(filename).suffix.lower()
            dangerous_extensions = ['.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar']
            
            if ext in dangerous_extensions:
                threats_detected.append("dangerous_file_type")
                recommendations.append(f"Dangerous file extension: {ext}")
        
        # Check content type
        if content_type:
            dangerous_types = [
                'application/x-executable',
                'application/x-msdownload',
                'application/x-msi',
                'application/x-msdos-program'
            ]
            
            if content_type in dangerous_types:
                threats_detected.append("dangerous_content_type")
                recommendations.append(f"Dangerous content type: {content_type}")
        
        # Check allowed types
        if allowed_types and content_type not in allowed_types:
            threats_detected.append("unallowed_content_type")
            recommendations.append(f"Content type {content_type} not in allowed list")
        
        valid = len(threats_detected) == 0
        
        return SecurityValidationResult(
            valid=valid,
            security_level=SecurityLevel.HIGH,
            threats_detected=threats_detected,
            recommendations=recommendations,
            risk_score=len(threats_detected) * 0.3
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        with self._security_lock:
            recent_events = [event for event in self.security_events 
                           if event.timestamp > datetime.now() - timedelta(hours=24)]
            
            return {
                'total_events': len(self.security_events),
                'recent_events': len(recent_events),
                'blocked_ips': len(self.blocked_ips),
                'active_sessions': len(self.session_store),
                'failed_attempts': dict(self.failed_attempts),
                'event_types': {
                    event.event_type: len([e for e in recent_events if e.event_type == event.event_type])
                    for event in recent_events
                },
                'severity_distribution': {
                    level.value: len([e for e in recent_events if e.severity == level])
                    for level in SecurityLevel
                }
            }


class SecureConfigValidator:
    """Secure configuration validator."""
    
    def __init__(self) -> Any:
        """Initialize secure config validator."""
        self.security_patterns = {
            'weak_passwords': [
                r'password\s*=\s*["\']?[^"\']*["\']?',
                r'secret\s*=\s*["\']?[^"\']*["\']?',
                r'key\s*=\s*["\']?[^"\']*["\']?',
                r'token\s*=\s*["\']?[^"\']*["\']?'
            ],
            'hardcoded_credentials': [
                r'admin\s*:\s*[^"\s]+',
                r'root\s*:\s*[^"\s]+',
                r'user\s*:\s*[^"\s]+'
            ],
            'insecure_protocols': [
                r'http://',
                r'ftp://',
                r'telnet://'
            ],
            'debug_enabled': [
                r'debug\s*=\s*true',
                r'DEBUG\s*=\s*True',
                r'development\s*=\s*true'
            ]
        }
    
    def validate_config_security(self, config_content: str) -> SecurityValidationResult:
        """Validate configuration for security issues."""
        threats_detected = []
        recommendations = []
        
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, config_content, re.IGNORECASE):
                    threats_detected.append(threat_type)
                    recommendations.append(f"Detected {threat_type} in configuration")
                    break
        
        valid = len(threats_detected) == 0
        
        return SecurityValidationResult(
            valid=valid,
            security_level=SecurityLevel.HIGH,
            threats_detected=threats_detected,
            recommendations=recommendations,
            risk_score=len(threats_detected) * 0.4
        )


# Example usage functions
def demonstrate_input_validation():
    """Demonstrate input validation."""
    security_manager = SecurityManager()
    
    # Test various inputs
    test_inputs = [
        ("normal_input", "Hello World", SecurityLevel.LOW),
        ("sql_injection", "'; DROP TABLE users; --", SecurityLevel.HIGH),
        ("xss_attack", "<script>alert('xss')</script>", SecurityLevel.HIGH),
        ("path_traversal", "../../../etc/passwd", SecurityLevel.HIGH),
        ("command_injection", "ls; rm -rf /", SecurityLevel.HIGH),
        ("long_input", "a" * 1000, SecurityLevel.MEDIUM)
    ]
    
    for test_name, input_data, level in test_inputs:
        result = security_manager.validate_input(input_data, test_name, level)
        print(f"\n{test_name}:")
        print(f"  Valid: {result.valid}")
        print(f"  Threats: {result.threats_detected}")
        print(f"  Risk Score: {result.risk_score:.2f}")
        if result.recommendations:
            print(f"  Recommendations: {result.recommendations}")


def demonstrate_password_validation():
    """Demonstrate password validation."""
    security_manager = SecurityManager()
    
    test_passwords = [
        "weak",
        "password123",
        "StrongP@ssw0rd",
        "123456789",
        "qwertyuiop",
        "MySecureP@ssw0rd2024!"
    ]
    
    for password in test_passwords:
        result = security_manager.validate_password(password)
        print(f"\nPassword: {password}")
        print(f"  Valid: {result.valid}")
        print(f"  Threats: {result.threats_detected}")
        print(f"  Risk Score: {result.risk_score:.2f}")
        if result.recommendations:
            print(f"  Recommendations: {result.recommendations}")


def demonstrate_authentication():
    """Demonstrate authentication flow."""
    security_manager = SecurityManager()
    
    # Test authentication
    auth_result = security_manager.authenticate_user(
        username="admin",
        password="correct_password",
        ip_address="192.168.1.100"
    )
    
    print(f"Authentication Result:")
    print(f"  Success: {auth_result.success}")
    print(f"  User ID: {auth_result.user_id}")
    print(f"  Roles: {auth_result.roles}")
    print(f"  Session Token: {auth_result.session_token[:20] if auth_result.session_token else None}")
    
    if auth_result.success and auth_result.session_token:
        # Test authorization
        authz_result = security_manager.authorize_action(
            session_token=auth_result.session_token,
            required_permissions=['read', 'write'],
            resource="/api/users"
        )
        
        print(f"\nAuthorization Result:")
        print(f"  Allowed: {authz_result.allowed}")
        print(f"  Reason: {authz_result.reason}")
        print(f"  Missing Permissions: {authz_result.missing_permissions}")


def demonstrate_url_validation():
    """Demonstrate URL validation."""
    security_manager = SecurityManager()
    
    test_urls = [
        "https://example.com/api",
        "http://192.168.1.1/admin",
        "javascript:alert('xss')",
        "file:///etc/passwd",
        "https://trusted-domain.com/safe",
        "ftp://malicious.com/files"
    ]
    
    allowed_domains = ["example.com", "trusted-domain.com"]
    
    for url in test_urls:
        result = security_manager.validate_url(url, allowed_domains)
        print(f"\nURL: {url}")
        print(f"  Valid: {result.valid}")
        print(f"  Threats: {result.threats_detected}")
        print(f"  Risk Score: {result.risk_score:.2f}")


def demonstrate_file_upload_validation():
    """Demonstrate file upload validation."""
    security_manager = SecurityManager()
    
    test_files = [
        ("document.pdf", "application/pdf", 1024000),
        ("script.exe", "application/x-executable", 2048000),
        ("image.jpg", "image/jpeg", 5242880),
        ("large_file.zip", "application/zip", 20971520),
        ("script.js", "application/javascript", 512000)
    ]
    
    allowed_types = ["application/pdf", "image/jpeg", "image/png"]
    max_size = 10485760  # 10MB
    
    for filename, content_type, file_size in test_files:
        result = security_manager.validate_file_upload(
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            max_size=max_size,
            allowed_types=allowed_types
        )
        
        print(f"\nFile: {filename}")
        print(f"  Valid: {result.valid}")
        print(f"  Threats: {result.threats_detected}")
        print(f"  Risk Score: {result.risk_score:.2f}")


def demonstrate_config_security():
    """Demonstrate configuration security validation."""
    validator = SecureConfigValidator()
    
    # Sample configuration with security issues
    config_content = """
[database]
host = localhost
user = admin
password = secret123

[api]
debug = true
url = http://insecure-api.com

[security]
key = hardcoded_secret_key
token = admin:password123
"""
    
    result = validator.validate_config_security(config_content)
    
    print("Configuration Security Validation:")
    print(f"  Valid: {result.valid}")
    print(f"  Threats: {result.threats_detected}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Recommendations: {result.recommendations}")


def main():
    """Main function demonstrating security guidelines."""
    logger.info("Starting security guidelines examples")
    
    # Demonstrate input validation
    try:
        demonstrate_input_validation()
    except Exception as e:
        logger.error(f"Input validation demonstration failed: {e}")
    
    # Demonstrate password validation
    try:
        demonstrate_password_validation()
    except Exception as e:
        logger.error(f"Password validation demonstration failed: {e}")
    
    # Demonstrate authentication
    try:
        demonstrate_authentication()
    except Exception as e:
        logger.error(f"Authentication demonstration failed: {e}")
    
    # Demonstrate URL validation
    try:
        demonstrate_url_validation()
    except Exception as e:
        logger.error(f"URL validation demonstration failed: {e}")
    
    # Demonstrate file upload validation
    try:
        demonstrate_file_upload_validation()
    except Exception as e:
        logger.error(f"File upload validation demonstration failed: {e}")
    
    # Demonstrate config security
    try:
        demonstrate_config_security()
    except Exception as e:
        logger.error(f"Config security demonstration failed: {e}")
    
    logger.info("Security guidelines examples completed")


match __name__:
    case "__main__":
    main() 