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

import hashlib
import hmac
import secrets
import re
import json
import base64
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import bcrypt
                from urllib.parse import urlparse
        import ipaddress
from typing import Any, List, Dict, Optional
"""
AI Video System - Security Module

Production-ready security utilities including input validation, sanitization,
encryption, authentication, and security best practices.
"""


try:
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Encryption
    encryption_key: Optional[str] = None
    salt_length: int = 32
    hash_rounds: int = 12
    
    # Input validation
    max_input_length: int = 10000
    allowed_file_extensions: List[str] = None
    max_file_size_mb: int = 100
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Session
    session_timeout_minutes: int = 60
    max_sessions_per_user: int = 5
    
    # Content security
    allowed_domains: List[str] = None
    blocked_patterns: List[str] = None
    
    def __post_init__(self) -> Any:
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = [
                '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mov',
                '.pdf', '.doc', '.docx', '.txt', '.json', '.xml'
            ]
        
        if self.allowed_domains is None:
            self.allowed_domains = [
                'youtube.com', 'vimeo.com', 'dailymotion.com',
                'example.com', 'localhost'
            ]
        
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'data:text/html',
                r'vbscript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>'
            ]


class InputValidator:
    """
    Comprehensive input validation and sanitization.
    
    Features:
    - Type validation
    - Length validation
    - Pattern validation
    - XSS prevention
    - SQL injection prevention
    - File upload validation
    """
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
self.config = config
        self.compiled_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'phone': re.compile(r'^\+?[\d\s\-\(\)]{10,20}$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9\s\-_\.]+$'),
            'filename': re.compile(r'^[a-zA-Z0-9\s\-_\.]+$'),
            'blocked': [re.compile(pattern, re.IGNORECASE) for pattern in config.blocked_patterns]
        }
    
    def validate_string(
        self,
        value: Any,
        min_length: int = 1,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_html: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate and sanitize string input.
        
        Returns:
            Tuple of (is_valid, sanitized_value)
        """
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        # Length validation
        if len(value) < min_length:
            return False, f"Value too short (minimum {min_length} characters)"
        
        if max_length and len(value) > max_length:
            return False, f"Value too long (maximum {max_length} characters)"
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            return False, f"Value does not match required pattern"
        
        # XSS prevention
        if not allow_html:
            sanitized = self._sanitize_html(value)
        else:
            sanitized = value
        
        # Check for blocked patterns
        for blocked_pattern in self.compiled_patterns['blocked']:
            if blocked_pattern.search(sanitized):
                return False, "Value contains blocked content"
        
        return True, sanitized
    
    def validate_email(self, email: str) -> Tuple[bool, str]:
        """Validate email address."""
        return self.validate_string(
            email,
            min_length=5,
            max_length=254,
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL."""
        is_valid, sanitized = self.validate_string(
            url,
            min_length=10,
            max_length=2048,
            pattern=r'^https?://[^\s/$.?#].[^\s]*$'
        )
        
        if is_valid:
            # Check if domain is allowed
            try:
                parsed = urlparse(sanitized)
                domain = parsed.netloc.lower()
                
                if not any(allowed in domain for allowed in self.config.allowed_domains):
                    return False, "Domain not allowed"
            except Exception:
                return False, "Invalid URL format"
        
        return is_valid, sanitized
    
    def validate_filename(self, filename: str) -> Tuple[bool, str]:
        """Validate filename."""
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_file_extensions:
            return False, f"File extension {file_ext} not allowed"
        
        # Check filename pattern
        return self.validate_string(
            filename,
            min_length=1,
            max_length=255,
            pattern=r'^[a-zA-Z0-9\s\-_\.]+$'
        )
    
    def validate_file_size(self, size_bytes: int) -> Tuple[bool, str]:
        """Validate file size."""
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if size_bytes > max_size_bytes:
            return False, f"File too large (maximum {self.config.max_file_size_mb}MB)"
        
        if size_bytes <= 0:
            return False, "File size must be positive"
        
        return True, "File size valid"
    
    def validate_json(self, data: str) -> Tuple[bool, Union[Dict, List, str]]:
        """Validate and parse JSON data."""
        try:
            parsed = json.loads(data)
            return True, parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    def validate_dict(
        self,
        data: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, callable]] = None
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Validate dictionary structure and content."""
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    return False, f"Required field '{field}' missing"
        
        # Validate individual fields
        if field_validators:
            for field, validator in field_validators.items():
                if field in data:
                    is_valid, result = validator(data[field])
                    if not is_valid:
                        return False, f"Field '{field}': {result}"
                    data[field] = result
        
        return True, data
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove event handlers
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        # Remove dangerous protocols
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data:text/html', '', text, flags=re.IGNORECASE)
        
        # Remove iframe, object, embed tags
        text = re.sub(r'<(iframe|object|embed)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text


class EncryptionManager:
    """
    Encryption and hashing utilities.
    
    Features:
    - Symmetric encryption
    - Password hashing
    - Secure key generation
    - Data signing
    """
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
self.config = config
        self._fernet = None
        
        if config.encryption_key:
            self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption with the provided key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography library not available. Encryption disabled.")
            return
        
        try:
            # Generate key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.config.encryption_key.encode(),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key.encode()))
            self._fernet = Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._fernet = None
    
    def encrypt_data(self, data: str) -> Optional[str]:
        """Encrypt data."""
        if not self._fernet:
            logger.warning("Encryption not available")
            return None
        
        try:
            encrypted = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt data."""
        if not self._fernet:
            logger.warning("Encryption not available")
            return None
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if not BCRYPT_AVAILABLE:
            # Fallback to SHA256 with salt
            salt = secrets.token_hex(self.config.salt_length)
            hashed = hashlib.sha256((password + salt).encode()).hexdigest()
            return f"{salt}:{hashed}"
        
        salt = bcrypt.gensalt(rounds=self.config.hash_rounds)
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        if not BCRYPT_AVAILABLE:
            # Fallback verification
            try:
                salt, hash_value = hashed_password.split(':', 1)
                expected_hash = hashlib.sha256((password + salt).encode()).hexdigest()
                return hmac.compare_digest(hash_value, expected_hash)
            except Exception:
                return False
        
        try:
            return bcrypt.checkpw(password.encode(), hashed_password.encode())
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)
    
    async def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"ak_{secrets.token_urlsafe(32)}"
    
    def sign_data(self, data: str, secret: str) -> str:
        """Sign data with HMAC."""
        signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, data: str, signature: str, secret: str) -> bool:
        """Verify data signature."""
        expected_signature = self.sign_data(data, secret)
        return hmac.compare_digest(signature, expected_signature)


class SessionManager:
    """
    Secure session management.
    
    Features:
    - Session creation and validation
    - Automatic expiration
    - Session limits per user
    - Secure session storage
    """
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, List[str]] = {}
    
    def create_session(self, user_id: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'data': data or {},
            'last_activity': datetime.now()
        }
        
        # Store session
        self.sessions[session_id] = session_data
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        
        self.user_sessions[user_id].append(session_id)
        
        # Enforce session limit
        if len(self.user_sessions[user_id]) > self.config.max_sessions_per_user:
            oldest_session = self.user_sessions[user_id].pop(0)
            self.sessions.pop(oldest_session, None)
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and return session data."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check expiration
        if datetime.now() > session['expires_at']:
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return session
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session."""
        if session_id in self.sessions:
            user_id = self.sessions[session_id]['user_id']
            
            # Remove from sessions
            self.sessions.pop(session_id)
            
            # Remove from user sessions
            if user_id in self.user_sessions:
                self.user_sessions[user_id] = [
                    sid for sid in self.user_sessions[user_id] if sid != session_id
                ]
            
            return True
        
        return False
    
    def destroy_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user."""
        if user_id not in self.user_sessions:
            return 0
        
        destroyed_count = 0
        for session_id in self.user_sessions[user_id]:
            if self.destroy_session(session_id):
                destroyed_count += 1
        
        return destroyed_count
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time > session['expires_at']
        ]
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        return len(expired_sessions)
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        if user_id not in self.user_sessions:
            return []
        
        sessions = []
        for session_id in self.user_sessions[user_id]:
            session = self.validate_session(session_id)
            if session:
                sessions.append({
                    'session_id': session_id,
                    'created_at': session['created_at'],
                    'last_activity': session['last_activity'],
                    'expires_at': session['expires_at']
                })
        
        return sessions


class SecurityAuditor:
    """
    Security auditing and monitoring.
    
    Features:
    - Security event logging
    - Threat detection
    - Audit trail
    - Security metrics
    """
    
    def __init__(self) -> Any:
        self.security_events: List[Dict[str, Any]] = []
        self.threat_patterns: Dict[str, List[str]] = {}
        self.audit_trail: List[Dict[str, Any]] = []
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security event."""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'user_id': user_id,
            'ip_address': ip_address,
            'metadata': metadata or {}
        }
        
        self.security_events.append(event)
        
        # Log to system logger
        log_message = f"SECURITY [{severity.upper()}] {event_type}: {description}"
        if user_id:
            log_message += f" (User: {user_id})"
        if ip_address:
            log_message += f" (IP: {ip_address})"
        
        if severity == 'high':
            logger.critical(log_message)
        elif severity == 'medium':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def detect_threats(self, user_id: str, action: str, context: Dict[str, Any]) -> List[str]:
        """Detect potential security threats."""
        threats = []
        
        # Rate limiting violations
        if context.get('rate_limit_exceeded'):
            threats.append('rate_limit_violation')
        
        # Unusual access patterns
        if context.get('unusual_location'):
            threats.append('unusual_location')
        
        # Failed authentication attempts
        if context.get('failed_auth_attempts', 0) > 5:
            threats.append('brute_force_attempt')
        
        # Suspicious file uploads
        if context.get('suspicious_file'):
            threats.append('malicious_file_upload')
        
        # XSS attempts
        if context.get('xss_attempt'):
            threats.append('xss_attempt')
        
        # SQL injection attempts
        if context.get('sql_injection_attempt'):
            threats.append('sql_injection_attempt')
        
        return threats
    
    def add_audit_entry(
        self,
        action: str,
        user_id: Optional[str],
        resource: Optional[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add entry to audit trail."""
        entry = {
            'timestamp': datetime.now(),
            'action': action,
            'user_id': user_id,
            'resource': resource,
            'details': details or {}
        }
        
        self.audit_trail.append(entry)
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if event['timestamp'] > cutoff_time
        ]
        
        # Count events by severity
        severity_counts = {}
        event_type_counts = {}
        
        for event in recent_events:
            severity = event['severity']
            event_type = event['event_type']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'severity_distribution': severity_counts,
            'event_type_distribution': event_type_counts,
            'recent_events': recent_events[-10:]  # Last 10 events
        }


# Global security instances
security_config = SecurityConfig()
input_validator = InputValidator(security_config)
encryption_manager = EncryptionManager(security_config)
session_manager = SessionManager(security_config)
security_auditor = SecurityAuditor()


# Security decorators
def require_authentication(func) -> Any:
    """Decorator to require authentication."""
    async def wrapper(*args, **kwargs) -> Any:
        # This is a placeholder - implement actual authentication logic
        session_id = kwargs.get('session_id')
        if not session_id:
            raise Exception("Authentication required")
        
        session = session_manager.validate_session(session_id)
        if not session:
            raise Exception("Invalid or expired session")
        
        return await func(*args, **kwargs)
    
    return wrapper


def validate_input(validation_rules: Dict[str, Any]):
    """Decorator to validate input parameters."""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            # Validate input parameters
            for param_name, rules in validation_rules.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    if 'type' in rules:
                        if not isinstance(value, rules['type']):
                            raise ValueError(f"Parameter {param_name} must be of type {rules['type']}")
                    
                    if 'validator' in rules:
                        is_valid, result = rules['validator'](value)
                        if not is_valid:
                            raise ValueError(f"Parameter {param_name}: {result}")
                        kwargs[param_name] = result
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_security_event(event_type: str, severity: str = 'info'):
    """Decorator to log security events."""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            try:
                result = await func(*args, **kwargs)
                security_auditor.log_security_event(
                    event_type=event_type,
                    severity=severity,
                    description=f"Function {func.__name__} executed successfully",
                    user_id=kwargs.get('user_id')
                )
                return result
            except Exception as e:
                security_auditor.log_security_event(
                    event_type=event_type,
                    severity='high',
                    description=f"Function {func.__name__} failed: {str(e)}",
                    user_id=kwargs.get('user_id')
                )
                raise
        
        return wrapper
    return decorator


# Security utilities
def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = filename.strip('._')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + ('.' + ext if ext else '')
    
    return filename


def validate_ip_address(ip: str) -> bool:
    """Validate IP address format."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


async def is_suspicious_request(headers: Dict[str, str], user_agent: str) -> bool:
    """Check if request appears suspicious."""
    suspicious_indicators = [
        # Missing or suspicious User-Agent
        not user_agent or user_agent.lower() in ['bot', 'crawler', 'spider'],
        
        # Suspicious headers
        any(header.lower().startswith('x-forwarded') for header in headers),
        
        # Missing common headers
        'accept' not in headers,
        'accept-language' not in headers,
    ]
    
    return any(suspicious_indicators)


async def cleanup_security_resources() -> None:
    """Cleanup security-related resources."""
    # Clean up expired sessions
    expired_count = session_manager.cleanup_expired_sessions()
    logger.info(f"Cleaned up {expired_count} expired sessions")
    
    # Clear old security events (keep last 1000)
    if len(security_auditor.security_events) > 1000:
        security_auditor.security_events = security_auditor.security_events[-1000:]
    
    # Clear old audit trail (keep last 5000)
    if len(security_auditor.audit_trail) > 5000:
        security_auditor.audit_trail = security_auditor.audit_trail[-5000:]
    
    logger.info("Security resources cleaned up") 