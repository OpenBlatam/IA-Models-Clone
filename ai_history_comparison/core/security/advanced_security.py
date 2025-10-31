"""
Advanced Security System - Enhanced Security and Threat Protection

This module provides advanced security capabilities including:
- Multi-factor authentication and authorization
- Advanced threat detection and prevention
- Security monitoring and incident response
- Data encryption and privacy protection
- Compliance and audit management
- Security analytics and reporting
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import jwt
import bcrypt
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import ipaddress
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import uuid

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDOS = "ddos"
    MALWARE = "malware"
    PHISHING = "phishing"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SAML = "saml"
    BIOMETRIC = "biometric"
    MFA = "mfa"
    SSO = "sso"

class AuthorizationLevel(Enum):
    """Authorization levels"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication settings
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    password_history_count: int = 5
    password_expiry_days: int = 90
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Token settings
    token_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    token_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # Encryption settings
    encryption_key: bytes = field(default_factory=lambda: Fernet.generate_key())
    encryption_algorithm: str = "AES-256-GCM"
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # Threat detection
    enable_threat_detection: bool = True
    threat_detection_sensitivity: SecurityLevel = SecurityLevel.MEDIUM
    auto_block_suspicious_ips: bool = True
    
    # Audit settings
    enable_audit_logging: bool = True
    audit_retention_days: int = 365
    
    # Compliance
    enable_gdpr_compliance: bool = True
    enable_hipaa_compliance: bool = False
    enable_sox_compliance: bool = False

@dataclass
class User:
    """User entity"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

@dataclass
class SecurityEvent:
    """Security event"""
    id: str
    event_type: str
    severity: SecurityLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False

@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    threat_type: ThreatType
    source_ip: str
    target_resource: str
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PasswordValidator:
    """Advanced password validation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        if self._is_common_password(password):
            errors.append("Password is too common and easily guessable")
        
        if self._has_sequential_chars(password):
            errors.append("Password contains sequential characters")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is common"""
        common_passwords = {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "1234567890", "abc123"
        }
        return password.lower() in common_passwords
    
    def _has_sequential_chars(self, password: str) -> bool:
        """Check for sequential characters"""
        sequences = ["123", "abc", "qwe", "asd", "zxc"]
        password_lower = password.lower()
        return any(seq in password_lower for seq in sequences)

class EncryptionManager:
    """Advanced encryption management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = Fernet(config.encryption_key)
        self._private_key = None
        self._public_key = None
        self._generate_key_pair()
    
    def _generate_key_pair(self) -> None:
        """Generate RSA key pair"""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self._public_key = self._private_key.public_key()
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data using Fernet"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using Fernet"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    def generate_hmac(self, data: str, key: str) -> str:
        """Generate HMAC signature"""
        return hmac.new(
            key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac(self, data: str, signature: str, key: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = self.generate_hmac(data, key)
        return hmac.compare_digest(signature, expected_signature)
    
    def sign_data(self, data: str) -> str:
        """Sign data with private key"""
        signature = self._private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify signature with public key"""
        try:
            signature_bytes = base64.b64decode(signature)
            self._public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

class TokenManager:
    """Advanced token management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
    
    def generate_access_token(self, user: User) -> str:
        """Generate JWT access token"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, self.config.token_secret_key, algorithm="HS256")
        
        # Store token metadata
        self.active_tokens[token] = {
            "user_id": user.id,
            "created_at": datetime.utcnow(),
            "expires_at": payload["exp"],
            "last_used": datetime.utcnow()
        }
        
        return token
    
    def generate_refresh_token(self, user: User) -> str:
        """Generate refresh token"""
        payload = {
            "user_id": user.id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=self.config.refresh_token_expiry_days),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, self.config.token_secret_key, algorithm="HS256")
        
        # Store refresh token metadata
        self.refresh_tokens[token] = {
            "user_id": user.id,
            "created_at": datetime.utcnow(),
            "expires_at": payload["exp"]
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode token"""
        try:
            payload = jwt.decode(token, self.config.token_secret_key, algorithms=["HS256"])
            
            # Check if token is in active tokens
            if token not in self.active_tokens:
                return None
            
            # Update last used timestamp
            self.active_tokens[token]["last_used"] = datetime.utcnow()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            # Remove expired token
            self.active_tokens.pop(token, None)
            return None
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke token"""
        if token in self.active_tokens:
            del self.active_tokens[token]
            return True
        return False
    
    def revoke_all_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user"""
        revoked_count = 0
        
        # Revoke access tokens
        tokens_to_remove = [
            token for token, metadata in self.active_tokens.items()
            if metadata["user_id"] == user_id
        ]
        
        for token in tokens_to_remove:
            del self.active_tokens[token]
            revoked_count += 1
        
        # Revoke refresh tokens
        refresh_tokens_to_remove = [
            token for token, metadata in self.refresh_tokens.items()
            if metadata["user_id"] == user_id
        ]
        
        for token in refresh_tokens_to_remove:
            del self.refresh_tokens[token]
            revoked_count += 1
        
        return revoked_count
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens"""
        current_time = datetime.utcnow()
        cleaned_count = 0
        
        # Clean access tokens
        expired_tokens = [
            token for token, metadata in self.active_tokens.items()
            if metadata["expires_at"] < current_time
        ]
        
        for token in expired_tokens:
            del self.active_tokens[token]
            cleaned_count += 1
        
        # Clean refresh tokens
        expired_refresh_tokens = [
            token for token, metadata in self.refresh_tokens.items()
            if metadata["expires_at"] < current_time
        ]
        
        for token in expired_refresh_tokens:
            del self.refresh_tokens[token]
            cleaned_count += 1
        
        return cleaned_count

class RateLimiter:
    """Advanced rate limiting"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        self.blocked_until: Dict[str, datetime] = {}
    
    def is_allowed(self, ip_address: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed"""
        current_time = time.time()
        
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            if ip_address in self.blocked_until:
                if datetime.utcnow() < self.blocked_until[ip_address]:
                    return False, {
                        "blocked": True,
                        "reason": "IP temporarily blocked",
                        "blocked_until": self.blocked_until[ip_address].isoformat()
                    }
                else:
                    # Unblock IP
                    self.blocked_ips.remove(ip_address)
                    del self.blocked_until[ip_address]
        
        # Initialize request count for IP
        if ip_address not in self.request_counts:
            self.request_counts[ip_address] = []
        
        # Clean old requests (older than 1 minute)
        self.request_counts[ip_address] = [
            req_time for req_time in self.request_counts[ip_address]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.request_counts[ip_address]) >= self.config.rate_limit_requests_per_minute:
            # Block IP temporarily
            self.blocked_ips.add(ip_address)
            self.blocked_until[ip_address] = datetime.utcnow() + timedelta(minutes=5)
            
            return False, {
                "blocked": True,
                "reason": "Rate limit exceeded",
                "blocked_until": self.blocked_until[ip_address].isoformat(),
                "request_count": len(self.request_counts[ip_address])
            }
        
        # Add current request
        self.request_counts[ip_address].append(current_time)
        
        return True, {
            "blocked": False,
            "request_count": len(self.request_counts[ip_address]),
            "limit": self.config.rate_limit_requests_per_minute
        }
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Manually unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.blocked_until.pop(ip_address, None)
            return True
        return False

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.threat_patterns = self._load_threat_patterns()
        self.suspicious_ips: Set[str] = set()
        self.threat_intelligence: List[ThreatIntelligence] = []
    
    def _load_threat_patterns(self) -> Dict[ThreatType, List[str]]:
        """Load threat detection patterns"""
        return {
            ThreatType.SQL_INJECTION: [
                r"('|(\\')|(;)|(\\;)|(--)|(\\*\\*)|(\\|\\|)|(\\+)|(\\-)|(\\*)|(\\/)|(\\%)|(\\^)|(\\&)|(\\|)|(\\~)|(\\!)|(\\<)|(\\>)|(\\=)|(\\?)|(\\:)|(\\;)|(\\,)|(\\.)|(\\[)|(\\])|(\\{)|(\\})|(\\(|\\))|(\\s)|(\\t)|(\\n)|(\\r)|(\\f)|(\\v)|(\\0)|(\\x00)|(\\x01)|(\\x02)|(\\x03)|(\\x04)|(\\x05)|(\\x06)|(\\x07)|(\\x08)|(\\x0b)|(\\x0c)|(\\x0e)|(\\x0f)|(\\x10)|(\\x11)|(\\x12)|(\\x13)|(\\x14)|(\\x15)|(\\x16)|(\\x17)|(\\x18)|(\\x19)|(\\x1a)|(\\x1b)|(\\x1c)|(\\x1d)|(\\x1e)|(\\x1f))",
                r"(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript|vbscript|onload|onerror|onclick)",
                r"(or|and)\\s+\\d+\\s*=\\s*\\d+",
                r"(or|and)\\s+['\"][^'\"]*['\"]\\s*=\\s*['\"][^'\"]*['\"]"
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\\w+\\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"<link[^>]*>",
                r"<meta[^>]*>",
                r"<style[^>]*>.*?</style>"
            ],
            ThreatType.CSRF: [
                r"<form[^>]*action[^>]*>",
                r"<input[^>]*type[^>]*hidden[^>]*>",
                r"document\\.cookie",
                r"document\\.location",
                r"window\\.location"
            ]
        }
    
    def detect_threat(self, request_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Detect potential threats in request data"""
        threats = []
        
        # Check for SQL injection
        sql_threats = self._detect_sql_injection(request_data)
        threats.extend(sql_threats)
        
        # Check for XSS
        xss_threats = self._detect_xss(request_data)
        threats.extend(xss_threats)
        
        # Check for CSRF
        csrf_threats = self._detect_csrf(request_data)
        threats.extend(csrf_threats)
        
        # Check for suspicious IP patterns
        ip_threats = self._detect_suspicious_ip(request_data)
        threats.extend(ip_threats)
        
        return threats
    
    def _detect_sql_injection(self, request_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Detect SQL injection attempts"""
        threats = []
        patterns = self.threat_patterns[ThreatType.SQL_INJECTION]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(ThreatIntelligence(
                            threat_type=ThreatType.SQL_INJECTION,
                            source_ip=request_data.get("ip_address", "unknown"),
                            target_resource=key,
                            confidence_score=0.8,
                            indicators=[pattern],
                            mitigation_actions=["Block request", "Log security event", "Alert administrators"]
                        ))
        
        return threats
    
    def _detect_xss(self, request_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Detect XSS attempts"""
        threats = []
        patterns = self.threat_patterns[ThreatType.XSS]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(ThreatIntelligence(
                            threat_type=ThreatType.XSS,
                            source_ip=request_data.get("ip_address", "unknown"),
                            target_resource=key,
                            confidence_score=0.9,
                            indicators=[pattern],
                            mitigation_actions=["Sanitize input", "Block request", "Log security event"]
                        ))
        
        return threats
    
    def _detect_csrf(self, request_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Detect CSRF attempts"""
        threats = []
        patterns = self.threat_patterns[ThreatType.CSRF]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(ThreatIntelligence(
                            threat_type=ThreatType.CSRF,
                            source_ip=request_data.get("ip_address", "unknown"),
                            target_resource=key,
                            confidence_score=0.7,
                            indicators=[pattern],
                            mitigation_actions=["Validate CSRF token", "Check referer header", "Log security event"]
                        ))
        
        return threats
    
    def _detect_suspicious_ip(self, request_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Detect suspicious IP patterns"""
        threats = []
        ip_address = request_data.get("ip_address")
        
        if ip_address:
            # Check if IP is in suspicious list
            if ip_address in self.suspicious_ips:
                threats.append(ThreatIntelligence(
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    source_ip=ip_address,
                    target_resource="system",
                    confidence_score=0.6,
                    indicators=["Known suspicious IP"],
                    mitigation_actions=["Block IP", "Monitor activity", "Alert administrators"]
                ))
            
            # Check for private/internal IPs accessing public endpoints
            try:
                ip_obj = ipaddress.ip_address(ip_address)
                if ip_obj.is_private and request_data.get("endpoint", "").startswith("/api/public"):
                    threats.append(ThreatIntelligence(
                        threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                        source_ip=ip_address,
                        target_resource=request_data.get("endpoint", ""),
                        confidence_score=0.5,
                        indicators=["Private IP accessing public endpoint"],
                        mitigation_actions=["Log access attempt", "Monitor for patterns"]
                    ))
            except ValueError:
                pass
        
        return threats

class SecurityAuditor:
    """Advanced security auditing system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_events: List[SecurityEvent] = []
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, List[str]]:
        """Load compliance rules"""
        return {
            "gdpr": [
                "Data processing must be lawful, fair and transparent",
                "Data must be collected for specified, explicit and legitimate purposes",
                "Data must be adequate, relevant and limited to what is necessary",
                "Data must be accurate and kept up to date",
                "Data must be kept in a form which permits identification for no longer than necessary",
                "Data must be processed in a manner that ensures appropriate security"
            ],
            "hipaa": [
                "Administrative safeguards must be implemented",
                "Physical safeguards must be implemented", 
                "Technical safeguards must be implemented",
                "Access controls must be implemented",
                "Audit controls must be implemented",
                "Integrity controls must be implemented"
            ],
            "sox": [
                "Internal controls over financial reporting must be implemented",
                "Management assessment of internal controls must be performed",
                "Independent auditor attestation must be obtained",
                "Disclosure controls and procedures must be implemented"
            ]
        }
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        self.audit_events.append(event)
        logger.warning(f"Security event logged: {event.event_type} - {event.description}")
    
    def get_audit_trail(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       event_type: Optional[str] = None,
                       user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Get audit trail with filters"""
        filtered_events = self.audit_events
        
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        return filtered_events
    
    def generate_compliance_report(self, compliance_standard: str) -> Dict[str, Any]:
        """Generate compliance report"""
        if compliance_standard not in self.compliance_rules:
            raise ValueError(f"Unknown compliance standard: {compliance_standard}")
        
        rules = self.compliance_rules[compliance_standard]
        compliance_status = {}
        
        for rule in rules:
            # This would implement actual compliance checking logic
            compliance_status[rule] = {
                "compliant": True,  # Placeholder
                "evidence": [],     # Placeholder
                "last_checked": datetime.utcnow()
            }
        
        return {
            "compliance_standard": compliance_standard,
            "report_date": datetime.utcnow(),
            "overall_compliance": True,  # Placeholder
            "rules": compliance_status
        }

class AdvancedSecurityManager:
    """Main advanced security management system"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.password_validator = PasswordValidator(self.config)
        self.encryption_manager = EncryptionManager(self.config)
        self.token_manager = TokenManager(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.threat_detector = ThreatDetector(self.config)
        self.security_auditor = SecurityAuditor(self.config)
        
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[SecurityEvent] = []
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task"""
        while True:
            try:
                # Clean up expired tokens
                cleaned_tokens = self.token_manager.cleanup_expired_tokens()
                if cleaned_tokens > 0:
                    logger.info(f"Cleaned up {cleaned_tokens} expired tokens")
                
                # Clean up old security events
                cutoff_date = datetime.utcnow() - timedelta(days=self.config.audit_retention_days)
                self.security_events = [
                    event for event in self.security_events
                    if event.timestamp > cutoff_date
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def authenticate_user(self, 
                              method: AuthenticationMethod,
                              credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
        """Authenticate user with multiple methods"""
        try:
            if method == AuthenticationMethod.PASSWORD:
                return await self._authenticate_password(credentials)
            elif method == AuthenticationMethod.TOKEN:
                return await self._authenticate_token(credentials)
            elif method == AuthenticationMethod.API_KEY:
                return await self._authenticate_api_key(credentials)
            elif method == AuthenticationMethod.MFA:
                return await self._authenticate_mfa(credentials)
            else:
                return False, None, "Unsupported authentication method"
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, None, "Authentication failed"
    
    async def _authenticate_password(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
        """Authenticate with password"""
        username = credentials.get("username")
        password = credentials.get("password")
        ip_address = credentials.get("ip_address", "unknown")
        
        if not username or not password:
            return False, None, "Username and password required"
        
        user = self.users.get(username)
        if not user:
            await self._log_security_event("failed_login", SecurityLevel.MEDIUM, None, ip_address, "Unknown user")
            return False, None, "Invalid credentials"
        
        # Check if user is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            await self._log_security_event("login_attempt_locked", SecurityLevel.HIGH, user.id, ip_address, "User account locked")
            return False, None, "Account is locked"
        
        # Check if user is active
        if not user.is_active:
            await self._log_security_event("login_attempt_inactive", SecurityLevel.MEDIUM, user.id, ip_address, "Inactive user account")
            return False, None, "Account is inactive"
        
        # Verify password
        if not self.encryption_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after max attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
                await self._log_security_event("account_locked", SecurityLevel.HIGH, user.id, ip_address, "Account locked due to failed attempts")
            
            await self._log_security_event("failed_login", SecurityLevel.MEDIUM, user.id, ip_address, "Invalid password")
            return False, None, "Invalid credentials"
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        await self._log_security_event("successful_login", SecurityLevel.LOW, user.id, ip_address, "Successful login")
        
        return True, user, None
    
    async def _authenticate_token(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
        """Authenticate with token"""
        token = credentials.get("token")
        ip_address = credentials.get("ip_address", "unknown")
        
        if not token:
            return False, None, "Token required"
        
        payload = self.token_manager.validate_token(token)
        if not payload:
            await self._log_security_event("invalid_token", SecurityLevel.MEDIUM, None, ip_address, "Invalid or expired token")
            return False, None, "Invalid or expired token"
        
        user_id = payload.get("user_id")
        user = self.users.get(user_id)
        
        if not user or not user.is_active:
            await self._log_security_event("token_user_inactive", SecurityLevel.MEDIUM, user_id, ip_address, "Token for inactive user")
            return False, None, "User account is inactive"
        
        return True, user, None
    
    async def _authenticate_api_key(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
        """Authenticate with API key"""
        api_key = credentials.get("api_key")
        ip_address = credentials.get("ip_address", "unknown")
        
        if not api_key:
            return False, None, "API key required"
        
        # This would implement API key validation logic
        # For now, return a placeholder
        await self._log_security_event("api_key_auth", SecurityLevel.LOW, None, ip_address, "API key authentication")
        return False, None, "API key authentication not implemented"
    
    async def _authenticate_mfa(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
        """Authenticate with multi-factor authentication"""
        username = credentials.get("username")
        password = credentials.get("password")
        mfa_code = credentials.get("mfa_code")
        ip_address = credentials.get("ip_address", "unknown")
        
        # First authenticate with password
        password_valid, user, error = await self._authenticate_password({
            "username": username,
            "password": password,
            "ip_address": ip_address
        })
        
        if not password_valid or not user:
            return False, None, error
        
        # Check if MFA is enabled
        if not user.mfa_enabled:
            return False, None, "MFA not enabled for user"
        
        # Validate MFA code (placeholder implementation)
        if not mfa_code or mfa_code != "123456":  # This would be replaced with actual MFA validation
            await self._log_security_event("mfa_failed", SecurityLevel.HIGH, user.id, ip_address, "Invalid MFA code")
            return False, None, "Invalid MFA code"
        
        await self._log_security_event("mfa_success", SecurityLevel.LOW, user.id, ip_address, "Successful MFA authentication")
        
        return True, user, None
    
    async def authorize_user(self, user: User, resource: str, action: AuthorizationLevel) -> bool:
        """Authorize user for resource and action"""
        # Check if user has required permissions
        required_permission = f"{resource}:{action.value}"
        
        if required_permission in user.permissions:
            return True
        
        # Check role-based permissions
        for role in user.roles:
            if self._role_has_permission(role, required_permission):
                return True
        
        await self._log_security_event("unauthorized_access", SecurityLevel.HIGH, user.id, "system", 
                                     f"Unauthorized access attempt: {required_permission}")
        
        return False
    
    def _role_has_permission(self, role: str, permission: str) -> bool:
        """Check if role has permission"""
        # This would implement role-based permission checking
        role_permissions = {
            "admin": ["*:*"],
            "user": ["read:*", "write:own_data"],
            "viewer": ["read:*"]
        }
        
        permissions = role_permissions.get(role, [])
        
        for perm in permissions:
            if perm == "*:*" or perm == permission:
                return True
            if perm.endswith("*") and permission.startswith(perm[:-1]):
                return True
        
        return False
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[ThreatIntelligence]]:
        """Validate request for security threats"""
        # Check rate limiting
        ip_address = request_data.get("ip_address", "unknown")
        allowed, rate_info = self.rate_limiter.is_allowed(ip_address)
        
        if not allowed:
            await self._log_security_event("rate_limit_exceeded", SecurityLevel.MEDIUM, None, ip_address, 
                                         f"Rate limit exceeded: {rate_info}")
            return False, []
        
        # Detect threats
        threats = self.threat_detector.detect_threat(request_data)
        
        # Log high-confidence threats
        for threat in threats:
            if threat.confidence_score > 0.8:
                await self._log_security_event("threat_detected", SecurityLevel.HIGH, None, threat.source_ip,
                                             f"Threat detected: {threat.threat_type.value}")
        
        return len(threats) == 0, threats
    
    async def _log_security_event(self, 
                                event_type: str,
                                severity: SecurityLevel,
                                user_id: Optional[str],
                                ip_address: str,
                                description: str,
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log security event"""
        event = SecurityEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=metadata.get("user_agent", "unknown") if metadata else "unknown",
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        self.security_auditor.log_security_event(event)
    
    def create_user(self, username: str, email: str, password: str, roles: List[str] = None) -> Tuple[bool, Optional[User], Optional[str]]:
        """Create new user"""
        # Validate password
        is_valid, errors = self.password_validator.validate_password(password)
        if not is_valid:
            return False, None, f"Password validation failed: {', '.join(errors)}"
        
        # Check if user already exists
        if username in self.users:
            return False, None, "Username already exists"
        
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=self.encryption_manager.hash_password(password),
            roles=roles or ["user"]
        )
        
        self.users[username] = user
        
        logger.info(f"User created: {username}")
        return True, user, None
    
    def generate_session_token(self, user: User) -> str:
        """Generate session token for user"""
        return self.token_manager.generate_access_token(user)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary"""
        return {
            "total_users": len(self.users),
            "active_sessions": len(self.active_sessions),
            "security_events_count": len(self.security_events),
            "blocked_ips": len(self.rate_limiter.blocked_ips),
            "active_tokens": len(self.token_manager.active_tokens),
            "threat_intelligence_count": len(self.threat_detector.threat_intelligence),
            "last_updated": datetime.utcnow().isoformat()
        }

# Global security manager instance
_global_security_manager: Optional[AdvancedSecurityManager] = None

def get_security_manager() -> AdvancedSecurityManager:
    """Get global security manager instance"""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = AdvancedSecurityManager()
    return _global_security_manager

async def authenticate_user(method: AuthenticationMethod, credentials: Dict[str, Any]) -> Tuple[bool, Optional[User], Optional[str]]:
    """Authenticate user using global security manager"""
    security_manager = get_security_manager()
    return await security_manager.authenticate_user(method, credentials)

async def authorize_user(user: User, resource: str, action: AuthorizationLevel) -> bool:
    """Authorize user using global security manager"""
    security_manager = get_security_manager()
    return await security_manager.authorize_user(user, resource, action)

async def validate_request(request_data: Dict[str, Any]) -> Tuple[bool, List[ThreatIntelligence]]:
    """Validate request using global security manager"""
    security_manager = get_security_manager()
    return await security_manager.validate_request(request_data)





















