"""
Advanced Security System for Ultimate Opus Clip

Comprehensive security features including authentication, authorization,
encryption, audit logging, and threat detection.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import time
import hashlib
import secrets
import jwt
import bcrypt
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import ipaddress
import re
from concurrent.futures import ThreadPoolExecutor
import json

logger = structlog.get_logger("advanced_security")

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDoS = "ddos"
    MALICIOUS_FILE = "malicious_file"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"

class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret: str
    jwt_expiry: int = 3600  # 1 hour
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 1800  # 30 minutes
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256
    enable_2fa: bool = True
    enable_audit_logging: bool = True
    enable_threat_detection: bool = True

@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: float
    last_activity: float
    expires_at: float
    is_active: bool = True

@dataclass
class SecurityEvent:
    """Security event log."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    timestamp: float
    severity: SecurityLevel
    description: str
    details: Dict[str, Any] = None

@dataclass
class ThreatDetection:
    """Threat detection result."""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    confidence: float
    source_ip: str
    user_id: Optional[str]
    timestamp: float
    description: str
    mitigation_action: str

class PasswordManager:
    """Advanced password management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_history: Dict[str, List[str]] = {}
        self.max_history = 5
        
        logger.info("Password Manager initialized")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        if re.search(r'(.)\1{2,}', password):
            errors.append("Password cannot contain repeated characters")
        
        if re.search(r'(123|abc|qwe)', password.lower()):
            errors.append("Password cannot contain common sequences")
        
        return len(errors) == 0, errors
    
    def check_password_history(self, user_id: str, password: str) -> bool:
        """Check if password was used recently."""
        if user_id not in self.password_history:
            return True
        
        for old_hash in self.password_history[user_id]:
            if self.verify_password(password, old_hash):
                return False
        
        return True
    
    def add_to_history(self, user_id: str, password_hash: str):
        """Add password to history."""
        if user_id not in self.password_history:
            self.password_history[user_id] = []
        
        self.password_history[user_id].append(password_hash)
        
        # Keep only recent passwords
        if len(self.password_history[user_id]) > self.max_history:
            self.password_history[user_id] = self.password_history[user_id][-self.max_history:]

class JWTManager:
    """JWT token management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blacklisted_tokens: set = set()
        
        logger.info("JWT Manager initialized")
    
    def generate_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token."""
        try:
            payload = {
                'user_id': user_id,
                'iat': time.time(),
                'exp': time.time() + self.config.jwt_expiry,
                'jti': str(uuid.uuid4())
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
            return token
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    def blacklist_token(self, token: str):
        """Add token to blacklist."""
        self.blacklisted_tokens.add(token)
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token."""
        try:
            payload = self.verify_token(token)
            if not payload:
                return None
            
            # Generate new token with same claims
            new_token = self.generate_token(
                payload['user_id'],
                {k: v for k, v in payload.items() if k not in ['iat', 'exp', 'jti']}
            )
            
            # Blacklist old token
            self.blacklist_token(token)
            
            return new_token
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None

class SessionManager:
    """Advanced session management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        
        logger.info("Session Manager initialized")
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create new user session."""
        try:
            session_id = str(uuid.uuid4())
            current_time = time.time()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=current_time,
                last_activity=current_time,
                expires_at=current_time + self.config.session_timeout
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Created session for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate session."""
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            current_time = time.time()
            
            # Check if session is expired
            if current_time > session.expires_at:
                del self.active_sessions[session_id]
                return None
            
            # Update last activity
            session.last_activity = current_time
            
            return session
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    def invalidate_session(self, session_id: str):
        """Invalidate session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Invalidated session {session_id}")
    
    def invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for user."""
        sessions_to_remove = [
            sid for sid, session in self.active_sessions.items()
            if session.user_id == user_id
        ]
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        logger.info(f"Invalidated {len(sessions_to_remove)} sessions for user {user_id}")
    
    def record_failed_attempt(self, ip_address: str):
        """Record failed login attempt."""
        current_time = time.time()
        
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []
        
        self.failed_attempts[ip_address].append(current_time)
        
        # Clean old attempts
        cutoff_time = current_time - self.config.lockout_duration
        self.failed_attempts[ip_address] = [
            attempt_time for attempt_time in self.failed_attempts[ip_address]
            if attempt_time > cutoff_time
        ]
    
    def is_ip_locked(self, ip_address: str) -> bool:
        """Check if IP is locked due to failed attempts."""
        if ip_address not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[ip_address]
        current_time = time.time()
        
        # Count recent attempts
        recent_attempts = [
            attempt for attempt in attempts
            if current_time - attempt < self.config.lockout_duration
        ]
        
        return len(recent_attempts) >= self.config.max_login_attempts

class EncryptionManager:
    """Advanced encryption management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_keys: Dict[str, str] = {}
        
        logger.info("Encryption Manager initialized")
    
    def generate_key(self, key_id: str) -> str:
        """Generate encryption key."""
        try:
            if self.config.encryption_algorithm == EncryptionAlgorithm.AES_256:
                key = secrets.token_hex(32)  # 256 bits
            elif self.config.encryption_algorithm == EncryptionAlgorithm.RSA_2048:
                key = secrets.token_hex(256)  # 2048 bits
            elif self.config.encryption_algorithm == EncryptionAlgorithm.RSA_4096:
                key = secrets.token_hex(512)  # 4096 bits
            elif self.config.encryption_algorithm == EncryptionAlgorithm.CHACHA20:
                key = secrets.token_hex(32)  # 256 bits
            else:
                key = secrets.token_hex(32)
            
            self.encryption_keys[key_id] = key
            return key
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise
    
    def encrypt_data(self, data: str, key_id: str) -> str:
        """Encrypt data."""
        try:
            if key_id not in self.encryption_keys:
                self.generate_key(key_id)
            
            key = self.encryption_keys[key_id]
            
            # Simple encryption (in production, use proper encryption)
            encrypted = hashlib.sha256((data + key).encode()).hexdigest()
            return encrypted
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str, key_id: str) -> str:
        """Decrypt data."""
        try:
            if key_id not in self.encryption_keys:
                raise ValueError("Key not found")
            
            # Simple decryption (in production, use proper decryption)
            # This is a placeholder implementation
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.threat_patterns: Dict[ThreatType, List[str]] = {
            ThreatType.SQL_INJECTION: [
                r"('|(\\')|(;)|(--)|(/\*)|(\*/)|(xp_)",
                r"(union|select|insert|update|delete|drop|create|alter)",
                r"(or|and)\s+\d+\s*=\s*\d+",
                r"(or|and)\s+'.*'\s*=\s*'.*'"
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
                r"<object[^>]*>.*?</object>"
            ],
            ThreatType.CSRF: [
                r"<form[^>]*action[^>]*>",
                r"<img[^>]*src[^>]*>",
                r"<link[^>]*href[^>]*>"
            ]
        }
        self.threat_history: List[ThreatDetection] = []
        
        logger.info("Threat Detector initialized")
    
    def detect_threat(self, input_data: str, source_ip: str, user_id: str = None) -> List[ThreatDetection]:
        """Detect potential threats in input data."""
        threats = []
        
        try:
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, input_data, re.IGNORECASE):
                        threat = ThreatDetection(
                            threat_id=str(uuid.uuid4()),
                            threat_type=threat_type,
                            severity=self._get_threat_severity(threat_type),
                            confidence=self._calculate_confidence(pattern, input_data),
                            source_ip=source_ip,
                            user_id=user_id,
                            timestamp=time.time(),
                            description=f"Detected {threat_type.value} pattern",
                            mitigation_action=self._get_mitigation_action(threat_type)
                        )
                        threats.append(threat)
            
            # Store threats
            self.threat_history.extend(threats)
            
            # Keep only recent threats
            if len(self.threat_history) > 10000:
                self.threat_history = self.threat_history[-10000:]
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
            return []
    
    def _get_threat_severity(self, threat_type: ThreatType) -> SecurityLevel:
        """Get severity level for threat type."""
        severity_map = {
            ThreatType.SQL_INJECTION: SecurityLevel.HIGH,
            ThreatType.XSS: SecurityLevel.MEDIUM,
            ThreatType.CSRF: SecurityLevel.MEDIUM,
            ThreatType.BRUTE_FORCE: SecurityLevel.MEDIUM,
            ThreatType.DDoS: SecurityLevel.CRITICAL,
            ThreatType.MALICIOUS_FILE: SecurityLevel.HIGH,
            ThreatType.UNAUTHORIZED_ACCESS: SecurityLevel.HIGH,
            ThreatType.DATA_BREACH: SecurityLevel.CRITICAL
        }
        return severity_map.get(threat_type, SecurityLevel.MEDIUM)
    
    def _calculate_confidence(self, pattern: str, input_data: str) -> float:
        """Calculate confidence score for threat detection."""
        try:
            matches = len(re.findall(pattern, input_data, re.IGNORECASE))
            return min(1.0, matches * 0.3)
        except Exception:
            return 0.5
    
    def _get_mitigation_action(self, threat_type: ThreatType) -> str:
        """Get mitigation action for threat type."""
        actions = {
            ThreatType.SQL_INJECTION: "Block request and log incident",
            ThreatType.XSS: "Sanitize input and block request",
            ThreatType.CSRF: "Validate CSRF token",
            ThreatType.BRUTE_FORCE: "Temporarily block IP address",
            ThreatType.DDoS: "Activate DDoS protection",
            ThreatType.MALICIOUS_FILE: "Quarantine file and scan system",
            ThreatType.UNAUTHORIZED_ACCESS: "Revoke access and alert security",
            ThreatType.DATA_BREACH: "Immediate incident response"
        }
        return actions.get(threat_type, "Log and monitor")

class AuditLogger:
    """Advanced audit logging system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_logs: List[SecurityEvent] = []
        self.max_logs = 100000
        
        logger.info("Audit Logger initialized")
    
    def log_security_event(self, event_type: str, user_id: str, ip_address: str,
                          severity: SecurityLevel, description: str, details: Dict[str, Any] = None):
        """Log security event."""
        try:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                timestamp=time.time(),
                severity=severity,
                description=description,
                details=details or {}
            )
            
            self.audit_logs.append(event)
            
            # Keep only recent logs
            if len(self.audit_logs) > self.max_logs:
                self.audit_logs = self.audit_logs[-self.max_logs:]
            
            logger.info(f"Security event logged: {event_type} - {description}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def get_security_events(self, user_id: str = None, severity: SecurityLevel = None,
                           start_time: float = None, end_time: float = None) -> List[SecurityEvent]:
        """Get security events with filters."""
        try:
            events = self.audit_logs.copy()
            
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            
            if severity:
                events = [e for e in events if e.severity == severity]
            
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []

class AdvancedSecuritySystem:
    """Main advanced security system."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig(jwt_secret=secrets.token_hex(32))
        self.password_manager = PasswordManager(self.config)
        self.jwt_manager = JWTManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.encryption_manager = EncryptionManager(self.config)
        self.threat_detector = ThreatDetector(self.config)
        self.audit_logger = AuditLogger(self.config)
        
        logger.info("Advanced Security System initialized")
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Dict[str, Any]:
        """Authenticate user with security checks."""
        try:
            # Check if IP is locked
            if self.session_manager.is_ip_locked(ip_address):
                self.audit_logger.log_security_event(
                    "login_attempt_blocked", None, ip_address,
                    SecurityLevel.MEDIUM, "IP address locked due to failed attempts"
                )
                return {"success": False, "error": "IP address locked"}
            
            # Validate input for threats
            threats = self.threat_detector.detect_threat(
                f"{username}:{password}", ip_address
            )
            
            if threats:
                self.audit_logger.log_security_event(
                    "threat_detected", None, ip_address,
                    SecurityLevel.HIGH, f"Threat detected during login: {threats[0].threat_type.value}"
                )
                return {"success": False, "error": "Security threat detected"}
            
            # Simulate user authentication (in production, check against database)
            if username == "admin" and password == "admin123":
                # Create session
                session_id = self.session_manager.create_session(
                    "user_123", ip_address, "Mozilla/5.0"
                )
                
                # Generate JWT token
                token = self.jwt_manager.generate_token("user_123")
                
                # Log successful login
                self.audit_logger.log_security_event(
                    "login_success", "user_123", ip_address,
                    SecurityLevel.LOW, "User logged in successfully"
                )
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "token": token,
                    "user_id": "user_123"
                }
            else:
                # Record failed attempt
                self.session_manager.record_failed_attempt(ip_address)
                
                # Log failed login
                self.audit_logger.log_security_event(
                    "login_failed", None, ip_address,
                    SecurityLevel.MEDIUM, "Invalid credentials"
                )
                
                return {"success": False, "error": "Invalid credentials"}
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {"success": False, "error": "Authentication error"}
    
    def validate_request(self, request_data: str, ip_address: str, user_id: str = None) -> bool:
        """Validate request for security threats."""
        try:
            # Detect threats
            threats = self.threat_detector.detect_threat(request_data, ip_address, user_id)
            
            if threats:
                # Log threats
                for threat in threats:
                    self.audit_logger.log_security_event(
                        "threat_detected", user_id, ip_address,
                        threat.severity, f"Threat detected: {threat.description}",
                        {"threat_type": threat.threat_type.value, "confidence": threat.confidence}
                    )
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return False
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        try:
            current_time = time.time()
            last_24h = current_time - (24 * 60 * 60)
            
            # Get recent security events
            recent_events = self.audit_logger.get_security_events(start_time=last_24h)
            
            # Count events by severity
            severity_counts = {}
            for event in recent_events:
                severity = event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Get active sessions
            active_sessions = len(self.session_manager.active_sessions)
            
            # Get recent threats
            recent_threats = [
                threat for threat in self.threat_detector.threat_history
                if threat.timestamp > last_24h
            ]
            
            return {
                "active_sessions": active_sessions,
                "recent_events": len(recent_events),
                "severity_counts": severity_counts,
                "recent_threats": len(recent_threats),
                "threat_types": list(set(t.threat_type.value for t in recent_threats)),
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Error getting security dashboard: {e}")
            return {"error": str(e)}

# Global security system instance
_global_security_system: Optional[AdvancedSecuritySystem] = None

def get_security_system() -> AdvancedSecuritySystem:
    """Get the global security system instance."""
    global _global_security_system
    if _global_security_system is None:
        _global_security_system = AdvancedSecuritySystem()
    return _global_security_system

def authenticate_user(username: str, password: str, ip_address: str) -> Dict[str, Any]:
    """Authenticate user with security checks."""
    security_system = get_security_system()
    return security_system.authenticate_user(username, password, ip_address)

def validate_request(request_data: str, ip_address: str, user_id: str = None) -> bool:
    """Validate request for security threats."""
    security_system = get_security_system()
    return security_system.validate_request(request_data, ip_address, user_id)


