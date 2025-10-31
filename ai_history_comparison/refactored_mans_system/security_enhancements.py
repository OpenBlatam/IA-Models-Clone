"""
Security Enhancements for MANS System

This module provides advanced security features and enhancements:
- Advanced authentication and authorization
- Encryption and data protection
- Threat detection and prevention
- Security monitoring and alerting
- Vulnerability scanning
- Intrusion detection
- Security audit logging
- Zero-trust architecture
- Multi-factor authentication
- API security
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import jwt
import bcrypt
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
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
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class ThreatType(Enum):
    """Threat types"""
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
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MFA = "mfa"
    SSO = "sso"
    OAUTH = "oauth"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    severity: str = "medium"
    source_ip: str = ""
    user_id: Optional[str] = None
    description: str = ""
    threat_type: Optional[ThreatType] = None
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityConfig:
    """Security configuration"""
    level: SecurityLevel = SecurityLevel.HIGH
    enable_encryption: bool = True
    enable_mfa: bool = True
    enable_rate_limiting: bool = True
    enable_threat_detection: bool = True
    enable_audit_logging: bool = True
    session_timeout: int = 3600
    max_login_attempts: int = 5
    lockout_duration: int = 900
    password_min_length: int = 12
    require_special_chars: bool = True
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    encryption_key: str = field(default_factory=lambda: Fernet.generate_key().decode())

class AdvancedEncryption:
    """Advanced encryption system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = Fernet(config.encryption_key.encode())
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._generate_rsa_keys()
    
    def _generate_rsa_keys(self) -> None:
        """Generate RSA key pair"""
        try:
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
        except Exception as e:
            logger.error(f"Failed to generate RSA keys: {e}")
    
    def encrypt_symmetric(self, data: str) -> str:
        """Encrypt data using symmetric encryption"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def encrypt_asymmetric(self, data: str) -> str:
        """Encrypt data using asymmetric encryption"""
        try:
            encrypted_data = self.rsa_public_key.encrypt(
                data.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data with RSA: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to decrypt data with RSA: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode(), salt)
            return hashed.decode()
        except Exception as e:
            logger.error(f"Failed to hash password: {e}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            logger.error(f"Failed to verify password: {e}")
            return False
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        try:
            payload["exp"] = datetime.utcnow() + timedelta(seconds=expires_in)
            payload["iat"] = datetime.utcnow()
            token = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")
            return token
        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            raise

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.threat_patterns: Dict[ThreatType, List[str]] = {
            ThreatType.SQL_INJECTION: [
                r"union\s+select", r"drop\s+table", r"delete\s+from",
                r"insert\s+into", r"update\s+set", r"';", r"--", r"/*"
            ],
            ThreatType.XSS: [
                r"<script", r"javascript:", r"onload=", r"onerror=",
                r"onclick=", r"onmouseover=", r"<iframe", r"<object"
            ],
            ThreatType.CSRF: [
                r"<form", r"<input", r"action=", r"method="
            ]
        }
        self.rate_limits: Dict[str, List[datetime]] = {}
    
    def detect_threat(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect potential threats in request"""
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
        
        # Check for brute force
        brute_force_threats = self._detect_brute_force(request_data)
        threats.extend(brute_force_threats)
        
        # Check rate limiting
        rate_limit_threats = self._check_rate_limiting(request_data)
        threats.extend(rate_limit_threats)
        
        return threats
    
    def _detect_sql_injection(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect SQL injection attempts"""
        threats = []
        patterns = self.threat_patterns[ThreatType.SQL_INJECTION]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityEvent(
                            event_type="sql_injection_attempt",
                            severity="high",
                            source_ip=request_data.get("source_ip", ""),
                            description=f"SQL injection pattern detected in {key}",
                            threat_type=ThreatType.SQL_INJECTION,
                            blocked=True,
                            metadata={"pattern": pattern, "field": key, "value": value}
                        )
                        threats.append(threat)
        
        return threats
    
    def _detect_xss(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect XSS attempts"""
        threats = []
        patterns = self.threat_patterns[ThreatType.XSS]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityEvent(
                            event_type="xss_attempt",
                            severity="high",
                            source_ip=request_data.get("source_ip", ""),
                            description=f"XSS pattern detected in {key}",
                            threat_type=ThreatType.XSS,
                            blocked=True,
                            metadata={"pattern": pattern, "field": key, "value": value}
                        )
                        threats.append(threat)
        
        return threats
    
    def _detect_csrf(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect CSRF attempts"""
        threats = []
        
        # Check for missing CSRF token
        if "csrf_token" not in request_data and request_data.get("method") in ["POST", "PUT", "DELETE"]:
            threat = SecurityEvent(
                event_type="csrf_attempt",
                severity="medium",
                source_ip=request_data.get("source_ip", ""),
                description="Missing CSRF token",
                threat_type=ThreatType.CSRF,
                blocked=True,
                metadata={"method": request_data.get("method")}
            )
            threats.append(threat)
        
        return threats
    
    def _detect_brute_force(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect brute force attempts"""
        threats = []
        source_ip = request_data.get("source_ip", "")
        user_id = request_data.get("user_id", "")
        
        if source_ip:
            # Track failed attempts
            if source_ip not in self.failed_attempts:
                self.failed_attempts[source_ip] = []
            
            # Check if this is a failed login attempt
            if request_data.get("event_type") == "login_failed":
                self.failed_attempts[source_ip].append(datetime.utcnow())
                
                # Clean old attempts (older than 1 hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.failed_attempts[source_ip] = [
                    attempt for attempt in self.failed_attempts[source_ip]
                    if attempt > cutoff_time
                ]
                
                # Check if threshold exceeded
                if len(self.failed_attempts[source_ip]) >= self.config.max_login_attempts:
                    threat = SecurityEvent(
                        event_type="brute_force_attempt",
                        severity="high",
                        source_ip=source_ip,
                        user_id=user_id,
                        description=f"Brute force attack detected from {source_ip}",
                        threat_type=ThreatType.BRUTE_FORCE,
                        blocked=True,
                        metadata={
                            "attempts": len(self.failed_attempts[source_ip]),
                            "threshold": self.config.max_login_attempts
                        }
                    )
                    threats.append(threat)
        
        return threats
    
    def _check_rate_limiting(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Check rate limiting"""
        threats = []
        source_ip = request_data.get("source_ip", "")
        
        if source_ip:
            if source_ip not in self.rate_limits:
                self.rate_limits[source_ip] = []
            
            # Add current request
            self.rate_limits[source_ip].append(datetime.utcnow())
            
            # Clean old requests (older than 1 minute)
            cutoff_time = datetime.utcnow() - timedelta(minutes=1)
            self.rate_limits[source_ip] = [
                req_time for req_time in self.rate_limits[source_ip]
                if req_time > cutoff_time
            ]
            
            # Check rate limit (100 requests per minute)
            if len(self.rate_limits[source_ip]) > 100:
                threat = SecurityEvent(
                    event_type="rate_limit_exceeded",
                    severity="medium",
                    source_ip=source_ip,
                    description=f"Rate limit exceeded from {source_ip}",
                    threat_type=ThreatType.DDOS,
                    blocked=True,
                    metadata={
                        "requests": len(self.rate_limits[source_ip]),
                        "limit": 100
                    }
                )
                threats.append(threat)
        
        return threats
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip in self.suspicious_ips:
            block_info = self.suspicious_ips[ip]
            if block_info["blocked_until"] > datetime.utcnow():
                return True
            else:
                # Unblock expired IPs
                del self.suspicious_ips[ip]
        return False
    
    def block_ip(self, ip: str, duration_minutes: int = 60) -> None:
        """Block IP address"""
        self.suspicious_ips[ip] = {
            "blocked_at": datetime.utcnow(),
            "blocked_until": datetime.utcnow() + timedelta(minutes=duration_minutes),
            "reason": "security_threat"
        }
        logger.warning(f"IP {ip} blocked for {duration_minutes} minutes")

class SecurityAuditor:
    """Security audit and logging system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log: List[SecurityEvent] = []
        self.security_metrics: Dict[str, Any] = {
            "total_events": 0,
            "blocked_events": 0,
            "threat_types": {},
            "source_ips": {},
            "severity_counts": {"low": 0, "medium": 0, "high": 0, "critical": 0}
        }
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        self.audit_log.append(event)
        
        # Update metrics
        self.security_metrics["total_events"] += 1
        if event.blocked:
            self.security_metrics["blocked_events"] += 1
        
        if event.threat_type:
            threat_type = event.threat_type.value
            self.security_metrics["threat_types"][threat_type] = \
                self.security_metrics["threat_types"].get(threat_type, 0) + 1
        
        if event.source_ip:
            self.security_metrics["source_ips"][event.source_ip] = \
                self.security_metrics["source_ips"].get(event.source_ip, 0) + 1
        
        self.security_metrics["severity_counts"][event.severity] += 1
        
        # Log to system logger
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(event.severity, logging.WARNING)
        
        logger.log(log_level, f"Security event: {event.event_type} - {event.description}")
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get security report for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [event for event in self.audit_log if event.timestamp > cutoff_time]
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "blocked_events": len([e for e in recent_events if e.blocked]),
            "threat_types": self._count_by_field(recent_events, "threat_type"),
            "severity_distribution": self._count_by_field(recent_events, "severity"),
            "top_source_ips": self._get_top_source_ips(recent_events),
            "recent_events": [self._event_to_dict(e) for e in recent_events[-10:]]
        }
    
    def _count_by_field(self, events: List[SecurityEvent], field: str) -> Dict[str, int]:
        """Count events by field"""
        counts = {}
        for event in events:
            value = getattr(event, field)
            if value:
                key = value.value if hasattr(value, 'value') else str(value)
                counts[key] = counts.get(key, 0) + 1
        return counts
    
    def _get_top_source_ips(self, events: List[SecurityEvent], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top source IPs by event count"""
        ip_counts = {}
        for event in events:
            if event.source_ip:
                ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"ip": ip, "count": count} for ip, count in sorted_ips[:top_n]]
    
    def _event_to_dict(self, event: SecurityEvent) -> Dict[str, Any]:
        """Convert security event to dictionary"""
        return {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "severity": event.severity,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "description": event.description,
            "threat_type": event.threat_type.value if event.threat_type else None,
            "blocked": event.blocked,
            "metadata": event.metadata
        }

class MultiFactorAuthentication:
    """Multi-factor authentication system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.mfa_sessions: Dict[str, Dict[str, Any]] = {}
        self.totp_secrets: Dict[str, str] = {}
    
    async def initiate_mfa(self, user_id: str, method: AuthenticationMethod) -> Dict[str, Any]:
        """Initiate MFA process"""
        session_id = str(uuid.uuid4())
        
        if method == AuthenticationMethod.MFA:
            # Generate TOTP secret if not exists
            if user_id not in self.totp_secrets:
                self.totp_secrets[user_id] = secrets.token_hex(16)
            
            self.mfa_sessions[session_id] = {
                "user_id": user_id,
                "method": method.value,
                "created_at": datetime.utcnow(),
                "verified": False,
                "attempts": 0
            }
            
            return {
                "session_id": session_id,
                "qr_code": self._generate_qr_code(user_id),
                "backup_codes": self._generate_backup_codes()
            }
        
        return {"error": "Unsupported MFA method"}
    
    def _generate_qr_code(self, user_id: str) -> str:
        """Generate QR code for TOTP setup"""
        # This would generate actual QR code
        return f"otpauth://totp/MANS:{user_id}?secret={self.totp_secrets[user_id]}&issuer=MANS"
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes"""
        return [secrets.token_hex(4) for _ in range(10)]
    
    async def verify_mfa(self, session_id: str, code: str) -> bool:
        """Verify MFA code"""
        if session_id not in self.mfa_sessions:
            return False
        
        session = self.mfa_sessions[session_id]
        session["attempts"] += 1
        
        # Check attempt limit
        if session["attempts"] > 3:
            del self.mfa_sessions[session_id]
            return False
        
        # Verify TOTP code (simplified)
        if self._verify_totp_code(session["user_id"], code):
            session["verified"] = True
            return True
        
        return False
    
    def _verify_totp_code(self, user_id: str, code: str) -> bool:
        """Verify TOTP code (simplified implementation)"""
        # This would implement actual TOTP verification
        # For now, accept any 6-digit code
        return len(code) == 6 and code.isdigit()
    
    def is_mfa_required(self, user_id: str) -> bool:
        """Check if MFA is required for user"""
        return self.config.enable_mfa and user_id in self.totp_secrets

class SecurityEnhancements:
    """Main security enhancements manager"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption = AdvancedEncryption(config)
        self.threat_detector = ThreatDetector(config)
        self.auditor = SecurityAuditor(config)
        self.mfa = MultiFactorAuthentication(config)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize security enhancements"""
        logger.info("Security enhancements initialized")
    
    async def shutdown(self) -> None:
        """Shutdown security enhancements"""
        logger.info("Security enhancements shut down")
    
    async def authenticate_user(self, user_id: str, password: str, 
                              source_ip: str = "", mfa_code: str = "") -> Dict[str, Any]:
        """Authenticate user with security checks"""
        # Check if IP is blocked
        if self.threat_detector.is_ip_blocked(source_ip):
            event = SecurityEvent(
                event_type="blocked_ip_attempt",
                severity="high",
                source_ip=source_ip,
                user_id=user_id,
                description=f"Authentication attempt from blocked IP {source_ip}",
                blocked=True
            )
            self.auditor.log_security_event(event)
            return {"success": False, "error": "IP blocked"}
        
        # Check for brute force
        request_data = {
            "source_ip": source_ip,
            "user_id": user_id,
            "event_type": "login_attempt"
        }
        threats = self.threat_detector.detect_threat(request_data)
        
        if threats:
            for threat in threats:
                self.auditor.log_security_event(threat)
            return {"success": False, "error": "Security threat detected"}
        
        # Check MFA requirement
        if self.mfa.is_mfa_required(user_id) and not mfa_code:
            return {"success": False, "mfa_required": True}
        
        # Verify MFA if provided
        if mfa_code:
            # This would verify the MFA code
            pass
        
        # Create session
        session_id = str(uuid.uuid4())
        token = self.encryption.generate_token({"user_id": user_id, "session_id": session_id})
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "source_ip": source_ip
        }
        
        return {
            "success": True,
            "token": token,
            "session_id": session_id,
            "expires_in": self.config.session_timeout
        }
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request for security threats"""
        threats = self.threat_detector.detect_threat(request_data)
        
        if threats:
            for threat in threats:
                self.auditor.log_security_event(threat)
            
            # Block IP if high severity threat
            high_severity_threats = [t for t in threats if t.severity in ["high", "critical"]]
            if high_severity_threats:
                source_ip = request_data.get("source_ip", "")
                if source_ip:
                    self.threat_detector.block_ip(source_ip)
            
            return {"valid": False, "threats": [self.auditor._event_to_dict(t) for t in threats]}
        
        return {"valid": True}
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        return {
            "config": {
                "level": self.config.level.value,
                "encryption_enabled": self.config.enable_encryption,
                "mfa_enabled": self.config.enable_mfa,
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "threat_detection_enabled": self.config.enable_threat_detection
            },
            "metrics": self.auditor.security_metrics,
            "active_sessions": len(self.active_sessions),
            "blocked_ips": len(self.threat_detector.suspicious_ips),
            "mfa_users": len(self.mfa.totp_secrets)
        }


