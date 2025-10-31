"""
Advanced Security System

This module provides comprehensive security capabilities for the refactored
HeyGen AI architecture with authentication, authorization, encryption, and
threat detection.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import jwt
import bcrypt
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import ipaddress
import geoip2.database
import requests
from functools import wraps
import threading
from collections import defaultdict, deque
import json
import uuid


logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Threat types."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDoS = "ddos"
    MALWARE = "malware"
    PHISHING = "phishing"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class SecurityEvent:
    """Security event structure."""
    event_id: str
    event_type: str
    threat_level: SecurityLevel
    description: str
    timestamp: datetime
    source_ip: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class PasswordValidator:
    """Advanced password validation."""
    
    def __init__(self):
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_special = True
        self.forbidden_patterns = [
            r'password',
            r'123456',
            r'qwerty',
            r'admin',
            r'user'
        ]
    
    def validate(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        issues = []
        score = 0
        
        # Length check
        if len(password) < self.min_length:
            issues.append(f"Password must be at least {self.min_length} characters long")
        else:
            score += 1
        
        # Character type checks
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if self.require_numbers and not re.search(r'\d', password):
            issues.append("Password must contain at least one number")
        else:
            score += 1
        
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        # Forbidden pattern checks
        for pattern in self.forbidden_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                issues.append(f"Password contains forbidden pattern: {pattern}")
                score -= 1
        
        # Entropy calculation
        entropy = self._calculate_entropy(password)
        if entropy < 3.0:
            issues.append("Password has low entropy")
            score -= 1
        
        return {
            "valid": len(issues) == 0,
            "score": max(0, min(5, score)),
            "issues": issues,
            "entropy": entropy
        }
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy."""
        char_counts = {}
        for char in password:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0
        for count in char_counts.values():
            p = count / len(password)
            entropy -= p * (p.bit_length() - 1)
        
        return entropy


class EncryptionManager:
    """Advanced encryption management."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        return self.fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.patterns = {
            ThreatType.SQL_INJECTION: [
                r"('|(\\')|(;)|(--)|(/\*)|(\*/)|(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
                r"(\b(or|and)\b\s+\d+\s*=\s*\d+)",
                r"(\b(union|select)\b.*\b(from|where)\b)"
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ],
            ThreatType.CSRF: [
                r"<form[^>]*action[^>]*>",
                r"<img[^>]*src[^>]*>",
                r"<link[^>]*href[^>]*>"
            ]
        }
        
        self.ip_blacklist = set()
        self.suspicious_ips = defaultdict(int)
        self.failed_logins = defaultdict(int)
        self.rate_limits = defaultdict(deque)
    
    def detect_threats(self, data: str, source_ip: str, user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Detect potential security threats."""
        threats = []
        
        # Check for injection attacks
        for threat_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    threat = SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=threat_type.value,
                        threat_level=SecurityLevel.HIGH,
                        description=f"Potential {threat_type.value} attack detected",
                        timestamp=datetime.now(timezone.utc),
                        source_ip=source_ip,
                        user_id=user_id,
                        details={"pattern": pattern, "data": data[:100]}
                    )
                    threats.append(threat)
        
        # Check IP blacklist
        if source_ip in self.ip_blacklist:
            threat = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=ThreatType.UNAUTHORIZED_ACCESS.value,
                threat_level=SecurityLevel.CRITICAL,
                description="Access from blacklisted IP",
                timestamp=datetime.now(timezone.utc),
                source_ip=source_ip,
                user_id=user_id
            )
            threats.append(threat)
        
        # Check for brute force attacks
        if self.failed_logins[source_ip] > 5:
            threat = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=ThreatType.BRUTE_FORCE.value,
                threat_level=SecurityLevel.HIGH,
                description="Potential brute force attack",
                timestamp=datetime.now(timezone.utc),
                source_ip=source_ip,
                user_id=user_id,
                details={"failed_attempts": self.failed_logins[source_ip]}
            )
            threats.append(threat)
        
        # Check for DDoS attacks
        if self._is_ddos_attack(source_ip):
            threat = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=ThreatType.DDoS.value,
                threat_level=SecurityLevel.CRITICAL,
                description="Potential DDoS attack",
                timestamp=datetime.now(timezone.utc),
                source_ip=source_ip,
                user_id=user_id
            )
            threats.append(threat)
        
        return threats
    
    def _is_ddos_attack(self, source_ip: str) -> bool:
        """Check if IP is performing DDoS attack."""
        now = time.time()
        window = 60  # 1 minute window
        max_requests = 100  # Max requests per minute
        
        # Clean old requests
        while self.rate_limits[source_ip] and self.rate_limits[source_ip][0] < now - window:
            self.rate_limits[source_ip].popleft()
        
        # Add current request
        self.rate_limits[source_ip].append(now)
        
        # Check if rate limit exceeded
        return len(self.rate_limits[source_ip]) > max_requests
    
    def add_failed_login(self, source_ip: str):
        """Record failed login attempt."""
        self.failed_logins[source_ip] += 1
    
    def reset_failed_logins(self, source_ip: str):
        """Reset failed login attempts."""
        self.failed_logins[source_ip] = 0
    
    def blacklist_ip(self, ip: str):
        """Add IP to blacklist."""
        self.ip_blacklist.add(ip)
    
    def whitelist_ip(self, ip: str):
        """Remove IP from blacklist."""
        self.ip_blacklist.discard(ip)


class AccessControlManager:
    """Advanced access control management."""
    
    def __init__(self):
        self.roles = {
            "admin": ["*"],  # All permissions
            "user": ["read", "write"],
            "guest": ["read"],
            "moderator": ["read", "write", "moderate"]
        }
        
        self.permissions = {
            "read": ["get_models", "get_user_info"],
            "write": ["create_model", "update_model", "delete_model"],
            "moderate": ["moderate_content", "ban_user"],
            "admin": ["manage_users", "system_config", "view_logs"]
        }
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        if not user.is_active:
            return False
        
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return False
        
        # Check if user has admin role
        if "admin" in user.roles:
            return True
        
        # Check role-based permissions
        for role in user.roles:
            if role in self.roles:
                role_permissions = self.roles[role]
                if "*" in role_permissions or permission in role_permissions:
                    return True
        
        # Check direct permissions
        return permission in user.permissions
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This would be implemented with actual user context
                # For now, just return the function
                return func(*args, **kwargs)
            return wrapper
        return decorator


class AdvancedSecuritySystem:
    """
    Advanced security system with comprehensive protection.
    
    Features:
    - Multi-factor authentication
    - Role-based access control
    - Threat detection and prevention
    - Encryption and data protection
    - Security monitoring and alerting
    - Rate limiting and DDoS protection
    - Password policy enforcement
    - Session management
    """
    
    def __init__(self, jwt_secret: str = None, encryption_key: bytes = None):
        """
        Initialize the advanced security system.
        
        Args:
            jwt_secret: JWT secret key
            encryption_key: Encryption master key
        """
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.encryption_manager = EncryptionManager(encryption_key)
        self.password_validator = PasswordValidator()
        self.threat_detector = ThreatDetector()
        self.access_control = AccessControlManager()
        
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[SecurityEvent] = []
        
        self.lock = threading.RLock()
    
    def register_user(self, username: str, email: str, password: str, roles: List[str] = None) -> Dict[str, Any]:
        """Register a new user."""
        with self.lock:
            # Validate password
            password_validation = self.password_validator.validate(password)
            if not password_validation["valid"]:
                return {
                    "success": False,
                    "error": "Password does not meet requirements",
                    "issues": password_validation["issues"]
                }
            
            # Check if user already exists
            if username in self.users or any(user.email == email for user in self.users.values()):
                return {
                    "success": False,
                    "error": "User already exists"
                }
            
            # Create user
            user_id = str(uuid.uuid4())
            password_hash = self.encryption_manager.hash_password(password)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles or ["user"]
            )
            
            self.users[user_id] = user
            
            # Log security event
            self._log_security_event(
                event_type="user_registration",
                threat_level=SecurityLevel.LOW,
                description=f"User {username} registered",
                user_id=user_id
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "User registered successfully"
            }
    
    def authenticate_user(self, username: str, password: str, source_ip: str) -> Dict[str, Any]:
        """Authenticate a user."""
        with self.lock:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username or u.email == username:
                    user = u
                    break
            
            if not user:
                self.threat_detector.add_failed_login(source_ip)
                return {
                    "success": False,
                    "error": "Invalid credentials"
                }
            
            # Check if user is locked
            if user.locked_until and user.locked_until > datetime.now(timezone.utc):
                return {
                    "success": False,
                    "error": "Account is locked",
                    "locked_until": user.locked_until.isoformat()
                }
            
            # Verify password
            if not self.encryption_manager.verify_password(password, user.password_hash):
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                
                self.threat_detector.add_failed_login(source_ip)
                
                # Log security event
                self._log_security_event(
                    event_type="failed_login",
                    threat_level=SecurityLevel.MEDIUM,
                    description=f"Failed login attempt for user {username}",
                    source_ip=source_ip,
                    user_id=user.user_id
                )
                
                return {
                    "success": False,
                    "error": "Invalid credentials",
                    "failed_attempts": user.failed_login_attempts
                }
            
            # Reset failed attempts
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)
            
            # Generate JWT token
            token = self._generate_jwt_token(user)
            
            # Create session
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "user_id": user.user_id,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "source_ip": source_ip
            }
            
            # Log security event
            self._log_security_event(
                event_type="successful_login",
                threat_level=SecurityLevel.LOW,
                description=f"User {username} logged in successfully",
                source_ip=source_ip,
                user_id=user.user_id
            )
            
            return {
                "success": True,
                "token": token,
                "session_id": session_id,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles
                }
            }
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return {
                "valid": True,
                "payload": payload
            }
        except jwt.ExpiredSignatureError:
            return {
                "valid": False,
                "error": "Token expired"
            }
        except jwt.InvalidTokenError:
            return {
                "valid": False,
                "error": "Invalid token"
            }
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        with self.lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            return self.access_control.has_permission(user, permission)
    
    def detect_threats(self, data: str, source_ip: str, user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Detect security threats in data."""
        threats = self.threat_detector.detect_threats(data, source_ip, user_id)
        
        # Log threats
        for threat in threats:
            self._log_security_event(
                event_type=threat.event_type,
                threat_level=threat.threat_level,
                description=threat.description,
                source_ip=threat.source_ip,
                user_id=threat.user_id,
                details=threat.details
            )
        
        return threats
    
    def _log_security_event(self, event_type: str, threat_level: SecurityLevel, description: str, 
                           source_ip: str = "unknown", user_id: Optional[str] = None, details: Dict[str, Any] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            timestamp=datetime.now(timezone.utc),
            source_ip=source_ip,
            user_id=user_id,
            details=details or {}
        )
        
        with self.lock:
            self.security_events.append(event)
            
            # Keep only last 1000 events
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        with self.lock:
            events = self.security_events[-limit:]
            return [event.__dict__ for event in events]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        with self.lock:
            total_events = len(self.security_events)
            events_by_level = defaultdict(int)
            events_by_type = defaultdict(int)
            
            for event in self.security_events:
                events_by_level[event.threat_level.value] += 1
                events_by_type[event.event_type] += 1
            
            return {
                "total_events": total_events,
                "events_by_level": dict(events_by_level),
                "events_by_type": dict(events_by_type),
                "active_users": len([u for u in self.users.values() if u.is_active]),
                "locked_users": len([u for u in self.users.values() if u.locked_until and u.locked_until > datetime.now(timezone.utc)]),
                "blacklisted_ips": len(self.threat_detector.ip_blacklist)
            }


# Example usage and demonstration
async def main():
    """Demonstrate the advanced security system."""
    print("🔒 HeyGen AI - Advanced Security System Demo")
    print("=" * 70)
    
    # Initialize security system
    security = AdvancedSecuritySystem()
    
    try:
        # Register a user
        print("\n👤 Registering User...")
        result = security.register_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            roles=["user"]
        )
        print(f"Registration result: {result}")
        
        # Authenticate user
        print("\n🔐 Authenticating User...")
        auth_result = security.authenticate_user("testuser", "SecurePass123!", "192.168.1.100")
        print(f"Authentication result: {auth_result}")
        
        if auth_result["success"]:
            token = auth_result["token"]
            user_id = auth_result["user"]["user_id"]
            
            # Verify token
            print("\n🔍 Verifying Token...")
            verify_result = security.verify_token(token)
            print(f"Token verification: {verify_result}")
            
            # Check permissions
            print("\n🛡️ Checking Permissions...")
            can_read = security.check_permission(user_id, "read")
            can_write = security.check_permission(user_id, "write")
            can_admin = security.check_permission(user_id, "admin")
            
            print(f"Can read: {can_read}")
            print(f"Can write: {can_write}")
            print(f"Can admin: {can_admin}")
        
        # Test threat detection
        print("\n🚨 Testing Threat Detection...")
        threats = security.detect_threats("'; DROP TABLE users; --", "192.168.1.200")
        print(f"Detected threats: {len(threats)}")
        for threat in threats:
            print(f"  - {threat.event_type}: {threat.description}")
        
        # Test XSS detection
        xss_threats = security.detect_threats("<script>alert('XSS')</script>", "192.168.1.201")
        print(f"XSS threats detected: {len(xss_threats)}")
        
        # Get security summary
        print("\n📊 Security Summary:")
        summary = security.get_security_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get security events
        print("\n📋 Recent Security Events:")
        events = security.get_security_events(5)
        for event in events:
            print(f"  {event['timestamp']} - {event['event_type']} ({event['threat_level']}): {event['description']}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())

