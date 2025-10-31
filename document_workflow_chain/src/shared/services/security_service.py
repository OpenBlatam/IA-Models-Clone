"""
Security Service
================

Advanced security service for authentication, authorization, and encryption.
"""

from __future__ import annotations
import asyncio
import logging
import hashlib
import secrets
import bcrypt
import jwt
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithm enumeration"""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    refresh_token_expiration_days: int = 30
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    encryption_key: Optional[str] = None
    enable_2fa: bool = True
    session_timeout_minutes: int = 60


@dataclass
class UserSession:
    """User session representation"""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event representation"""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: SecurityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


class SecurityService:
    """Advanced security service"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "default-secret-key"),
            encryption_key=os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        )
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.security_events: List[SecurityEvent] = []
        self.encryption_key = Fernet(self.config.encryption_key.encode())
        self.rsa_private_key: Optional[rsa.RSAPrivateKey] = None
        self.rsa_public_key: Optional[rsa.RSAPublicKey] = None
        self._generate_rsa_keys()
    
    def _generate_rsa_keys(self):
        """Generate RSA key pair"""
        try:
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            logger.info("RSA key pair generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate RSA keys: {e}")
    
    # Password Management
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate secure random password"""
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        password = ''.join(secrets.choice(characters) for _ in range(length))
        return password
    
    # JWT Token Management
    def create_access_token(self, user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT access token"""
        now = DateTimeHelpers.now_utc()
        expiration = now + timedelta(hours=self.config.jwt_expiration_hours)
        
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": expiration,
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = DateTimeHelpers.now_utc()
        expiration = now + timedelta(days=self.config.refresh_token_expiration_days)
        
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": expiration,
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("user_id")
        if not user_id:
            return None
        
        return self.create_access_token(user_id)
    
    # Session Management
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        now = DateTimeHelpers.now_utc()
        
        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = session
        
        # Log security event
        self._log_security_event(
            "session_created",
            user_id,
            ip_address,
            user_agent,
            SecurityLevel.MEDIUM,
            {"session_id": session_id}
        )
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str, user_agent: str) -> bool:
        """Validate user session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check if session is active
        if not session.is_active:
            return False
        
        # Check session timeout
        timeout_threshold = DateTimeHelpers.now_utc() - timedelta(minutes=self.config.session_timeout_minutes)
        if session.last_activity < timeout_threshold:
            self.invalidate_session(session_id)
            return False
        
        # Update last activity
        session.last_activity = DateTimeHelpers.now_utc()
        
        # Log security event
        self._log_security_event(
            "session_validated",
            session.user_id,
            ip_address,
            user_agent,
            SecurityLevel.LOW,
            {"session_id": session_id}
        )
        
        return True
    
    def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            
            # Log security event
            self._log_security_event(
                "session_invalidated",
                session.user_id,
                session.ip_address,
                session.user_agent,
                SecurityLevel.MEDIUM,
                {"session_id": session_id}
            )
            
            del self.active_sessions[session_id]
    
    def invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user"""
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.invalidate_session(session_id)
    
    # Login Attempt Management
    def record_failed_login(self, identifier: str, ip_address: str, user_agent: str):
        """Record failed login attempt"""
        now = DateTimeHelpers.now_utc()
        
        if identifier not in self.failed_login_attempts:
            self.failed_login_attempts[identifier] = []
        
        self.failed_login_attempts[identifier].append(now)
        
        # Clean old attempts
        cutoff_time = now - timedelta(minutes=self.config.lockout_duration_minutes)
        self.failed_login_attempts[identifier] = [
            attempt for attempt in self.failed_login_attempts[identifier]
            if attempt > cutoff_time
        ]
        
        # Log security event
        self._log_security_event(
            "failed_login",
            identifier,
            ip_address,
            user_agent,
            SecurityLevel.HIGH,
            {"attempt_count": len(self.failed_login_attempts[identifier])},
            success=False
        )
    
    def is_account_locked(self, identifier: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if identifier not in self.failed_login_attempts:
            return False
        
        attempts = self.failed_login_attempts[identifier]
        return len(attempts) >= self.config.max_login_attempts
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed login attempts for identifier"""
        if identifier in self.failed_login_attempts:
            del self.failed_login_attempts[identifier]
    
    # Encryption/Decryption
    def encrypt_data(self, data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Encrypt data"""
        try:
            if algorithm == EncryptionAlgorithm.AES_256:
                encrypted_data = self.encryption_key.encrypt(data.encode())
                return base64.b64encode(encrypted_data).decode()
            elif algorithm == EncryptionAlgorithm.RSA_2048:
                if not self.rsa_public_key:
                    raise ValueError("RSA public key not available")
                
                encrypted_data = self.rsa_public_key.encrypt(
                    data.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.b64encode(encrypted_data).decode()
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Decrypt data"""
        try:
            if algorithm == EncryptionAlgorithm.AES_256:
                decoded_data = base64.b64decode(encrypted_data.encode())
                decrypted_data = self.encryption_key.decrypt(decoded_data)
                return decrypted_data.decode()
            elif algorithm == EncryptionAlgorithm.RSA_2048:
                if not self.rsa_private_key:
                    raise ValueError("RSA private key not available")
                
                decoded_data = base64.b64decode(encrypted_data.encode())
                decrypted_data = self.rsa_private_key.decrypt(
                    decoded_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted_data.decode()
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    # API Key Management
    def generate_api_key(self, user_id: str, name: str) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Log security event
        self._log_security_event(
            "api_key_generated",
            user_id,
            "system",
            "system",
            SecurityLevel.MEDIUM,
            {"api_key_name": name, "hashed_key": hashed_key}
        )
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key (simplified implementation)"""
        # In a real implementation, you would check against stored hashed keys
        return len(api_key) >= 32
    
    # Security Event Logging
    def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        severity: SecurityLevel,
        details: Dict[str, Any],
        success: bool = True
    ):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=DateTimeHelpers.now_utc(),
            severity=severity,
            details=details,
            success=success
        )
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log to logger
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"Security event: {event_type} - User: {user_id} - IP: {ip_address} - Success: {success}"
        )
    
    def get_security_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[SecurityLevel] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get security events with filters"""
        events = self.security_events
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return events[-limit:]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        total_events = len(self.security_events)
        failed_events = len([e for e in self.security_events if not e.success])
        events_by_type = {}
        events_by_severity = {}
        
        for event in self.security_events:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            events_by_severity[event.severity.value] = events_by_severity.get(event.severity.value, 0) + 1
        
        return {
            "total_events": total_events,
            "failed_events": failed_events,
            "success_rate": (total_events - failed_events) / total_events if total_events > 0 else 0,
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "active_sessions": len(self.active_sessions),
            "locked_accounts": len(self.failed_login_attempts),
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
    
    # Cleanup
    def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        now = DateTimeHelpers.now_utc()
        timeout_threshold = now - timedelta(minutes=self.config.session_timeout_minutes)
        
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity < timeout_threshold:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def cleanup_old_security_events(self, days: int = 30):
        """Cleanup old security events"""
        cutoff_time = DateTimeHelpers.now_utc() - timedelta(days=days)
        self.security_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        logger.info(f"Cleaned up security events older than {days} days")


# Global security service
security_service = SecurityService()


# Utility functions
def hash_password(password: str) -> str:
    """Hash password"""
    return security_service.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password"""
    return security_service.verify_password(password, hashed_password)


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """Validate password strength"""
    return security_service.validate_password_strength(password)


def generate_secure_password(length: int = 16) -> str:
    """Generate secure password"""
    return security_service.generate_secure_password(length)


def create_access_token(user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
    """Create access token"""
    return security_service.create_access_token(user_id, additional_claims)


def create_refresh_token(user_id: str) -> str:
    """Create refresh token"""
    return security_service.create_refresh_token(user_id)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token"""
    return security_service.verify_token(token)


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Refresh access token"""
    return security_service.refresh_access_token(refresh_token)


def create_session(user_id: str, ip_address: str, user_agent: str) -> str:
    """Create session"""
    return security_service.create_session(user_id, ip_address, user_agent)


def validate_session(session_id: str, ip_address: str, user_agent: str) -> bool:
    """Validate session"""
    return security_service.validate_session(session_id, ip_address, user_agent)


def invalidate_session(session_id: str):
    """Invalidate session"""
    security_service.invalidate_session(session_id)


def record_failed_login(identifier: str, ip_address: str, user_agent: str):
    """Record failed login"""
    security_service.record_failed_login(identifier, ip_address, user_agent)


def is_account_locked(identifier: str) -> bool:
    """Check if account is locked"""
    return security_service.is_account_locked(identifier)


def encrypt_data(data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
    """Encrypt data"""
    return security_service.encrypt_data(data, algorithm)


def decrypt_data(encrypted_data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
    """Decrypt data"""
    return security_service.decrypt_data(encrypted_data, algorithm)


def generate_api_key(user_id: str, name: str) -> str:
    """Generate API key"""
    return security_service.generate_api_key(user_id, name)


def get_security_events(
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    severity: Optional[SecurityLevel] = None,
    limit: int = 100
) -> List[SecurityEvent]:
    """Get security events"""
    return security_service.get_security_events(user_id, event_type, severity, limit)


def get_security_stats() -> Dict[str, Any]:
    """Get security statistics"""
    return security_service.get_security_stats()




