#!/usr/bin/env python3
"""
Advanced Security Manager for Enhanced HeyGen AI
Provides enhanced security features including threat detection, encryption, access control, and security monitoring.
"""

import asyncio
import time
import json
import hashlib
import hmac
import secrets
import base64
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import jwt
from datetime import datetime, timedelta
import ipaddress
import re
import ssl
import socket
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

logger = structlog.get_logger()

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_IP = "suspicious_ip"
    MALFORMED_REQUEST = "malformed_request"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    DDoS_ATTEMPT = "ddos_attempt"

class SecurityEventType(Enum):
    """Types of security events."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    THREAT_DETECTED = "threat_detected"
    THREAT_BLOCKED = "threat_blocked"
    SECURITY_ALERT = "security_alert"
    SYSTEM_COMPROMISE = "system_compromise"

@dataclass
class SecurityEvent:
    """Represents a security event."""
    id: str
    event_type: SecurityEventType
    timestamp: float
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: SecurityLevel
    source: str
    session_id: Optional[str] = None

@dataclass
class Threat:
    """Represents a detected threat."""
    id: str
    threat_type: ThreatType
    timestamp: float
    ip_address: str
    user_id: Optional[str]
    user_agent: str
    details: Dict[str, Any]
    severity: SecurityLevel
    blocked: bool
    false_positive: bool = False
    resolved: bool = False

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    description: str
    enabled: bool
    rules: List[Dict[str, Any]]
    actions: List[str]
    priority: int
    created_at: float
    updated_at: float

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
    is_active: bool
    permissions: List[str]
    security_level: SecurityLevel

class AdvancedSecurityManager:
    """Advanced security manager with comprehensive threat detection and prevention."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        encryption_key: Optional[str] = None,
        enable_threat_detection: bool = True,
        enable_encryption: bool = True,
        enable_session_management: bool = True,
        max_login_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
        session_timeout: int = 3600,  # 1 hour
        max_concurrent_sessions: int = 3
    ):
        # Security keys
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.encryption_key = encryption_key or Fernet.generate_key()
        
        # Configuration
        self.enable_threat_detection = enable_threat_detection
        self.enable_encryption = enable_encryption
        self.enable_session_management = enable_session_management
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = lockout_duration
        self.session_timeout = session_timeout
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Encryption
        self.fernet = Fernet(self.encryption_key)
        
        # Storage
        self.security_events: List[SecurityEvent] = []
        self.threats: List[Threat] = []
        self.blocked_ips: Dict[str, float] = {}
        self.user_sessions: Dict[str, UserSession] = {}
        self.login_attempts: Dict[str, List[float]] = {}
        self.security_policies: List[SecurityPolicy] = []
        
        # Threat detection patterns
        self.threat_patterns = self._initialize_threat_patterns()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize security policies
        self._initialize_security_policies()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize threat detection patterns."""
        patterns = {
            "sql_injection": [
                re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter)\b)", re.IGNORECASE),
                re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
                re.compile(r"(\b(and|or)\s+\d+\s*=\s*\d+)", re.IGNORECASE)
            ],
            "xss": [
                re.compile(r"(<script[^>]*>.*?</script>)", re.IGNORECASE),
                re.compile(r"(javascript:)", re.IGNORECASE),
                re.compile(r"(on\w+\s*=)", re.IGNORECASE)
            ],
            "path_traversal": [
                re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
                re.compile(r"(/%2e%2e/|%2e%2e/)", re.IGNORECASE)
            ],
            "command_injection": [
                re.compile(r"(\b(cat|ls|pwd|whoami|id|uname)\b)", re.IGNORECASE),
                re.compile(r"(\$\(|`.*`|;.*)", re.IGNORECASE)
            ]
        }
        return patterns
    
    def _initialize_security_policies(self):
        """Initialize default security policies."""
        default_policies = [
            SecurityPolicy(
                name="brute_force_protection",
                description="Protect against brute force attacks",
                enabled=True,
                rules=[
                    {"type": "login_attempts", "max_attempts": self.max_login_attempts, "window": 300}
                ],
                actions=["block_ip", "alert_admin"],
                priority=1,
                created_at=time.time(),
                updated_at=time.time()
            ),
            SecurityPolicy(
                name="rate_limiting",
                description="Rate limiting for API endpoints",
                enabled=True,
                rules=[
                    {"type": "rate_limit", "max_requests": 100, "window": 60}
                ],
                actions=["throttle", "block_ip"],
                priority=2,
                created_at=time.time(),
                updated_at=time.time()
            ),
            SecurityPolicy(
                name="input_validation",
                description="Validate and sanitize input",
                enabled=True,
                rules=[
                    {"type": "pattern_matching", "patterns": list(self.threat_patterns.keys())}
                ],
                actions=["block_request", "log_threat"],
                priority=3,
                created_at=time.time(),
                updated_at=time.time()
            )
        ]
        
        self.security_policies = default_policies
    
    def _start_background_tasks(self):
        """Start background tasks."""
        self.monitoring_task = asyncio.create_task(self._security_monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _security_monitoring_loop(self):
        """Security monitoring loop."""
        while True:
            try:
                await self._analyze_security_events()
                await self._check_threat_patterns()
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup loop for old data."""
        while True:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cleanup(self):
        """Perform cleanup operations."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours
            
            # Clean up old security events
            self.security_events = [
                event for event in self.security_events
                if event.timestamp > cutoff_time
            ]
            
            # Clean up old threats
            self.threats = [
                threat for threat in self.threats
                if threat.timestamp > cutoff_time
            ]
            
            # Clean up expired sessions
            expired_sessions = [
                session_id for session_id, session in self.user_sessions.items()
                if session.expires_at < current_time
            ]
            for session_id in expired_sessions:
                del self.user_sessions[session_id]
            
            # Clean up old login attempts
            old_attempts = []
            for ip, attempts in self.login_attempts.items():
                self.login_attempts[ip] = [
                    attempt for attempt in attempts
                    if current_time - attempt < self.lockout_duration
                ]
                if not self.login_attempts[ip]:
                    old_attempts.append(ip)
            
            for ip in old_attempts:
                del self.login_attempts[ip]
            
            # Clean up expired IP blocks
            expired_blocks = [
                ip for ip, block_until in self.blocked_ips.items()
                if block_until < current_time
            ]
            for ip in expired_blocks:
                del self.blocked_ips[ip]
            
            logger.debug("Security cleanup completed")
            
        except Exception as e:
            logger.error(f"Security cleanup failed: {e}")
    
    async def _analyze_security_events(self):
        """Analyze recent security events for patterns."""
        try:
            current_time = time.time()
            recent_events = [
                event for event in self.security_events
                if current_time - event.timestamp < 300  # Last 5 minutes
            ]
            
            # Analyze login attempts
            login_events = [e for e in recent_events if e.event_type == SecurityEventType.LOGIN_ATTEMPT]
            for event in login_events:
                await self._analyze_login_attempt(event)
            
            # Analyze access patterns
            access_events = [e for e in recent_events if e.event_type in [SecurityEventType.ACCESS_GRANTED, SecurityEventType.ACCESS_DENIED]]
            await self._analyze_access_patterns(access_events)
            
        except Exception as e:
            logger.error(f"Security event analysis failed: {e}")
    
    async def _analyze_login_attempt(self, event: SecurityEvent):
        """Analyze a login attempt for suspicious activity."""
        try:
            ip = event.ip_address
            user_id = event.details.get("username", "unknown")
            
            # Track login attempts
            if ip not in self.login_attempts:
                self.login_attempts[ip] = []
            
            self.login_attempts[ip].append(event.timestamp)
            
            # Check if IP should be blocked
            recent_attempts = [
                attempt for attempt in self.login_attempts[ip]
                if time.time() - attempt < 300  # Last 5 minutes
            ]
            
            if len(recent_attempts) >= self.max_login_attempts:
                # Block IP
                self.blocked_ips[ip] = time.time() + self.lockout_duration
                
                # Create threat record
                threat = Threat(
                    id=secrets.token_urlsafe(16),
                    threat_type=ThreatType.BRUTE_FORCE,
                    timestamp=time.time(),
                    ip_address=ip,
                    user_id=user_id,
                    user_agent=event.user_agent,
                    details={
                        "attempts": len(recent_attempts),
                        "window": 300,
                        "blocked_until": self.blocked_ips[ip]
                    },
                    severity=SecurityLevel.HIGH,
                    blocked=True
                )
                
                self.threats.append(threat)
                
                # Log security event
                await self.log_security_event(
                    SecurityEventType.THREAT_BLOCKED,
                    user_id,
                    ip,
                    event.user_agent,
                    {"threat_id": threat.id, "reason": "brute_force_detected"},
                    SecurityLevel.HIGH
                )
                
                logger.warning(f"IP {ip} blocked due to brute force attempts")
                
        except Exception as e:
            logger.error(f"Login attempt analysis failed: {e}")
    
    async def _analyze_access_patterns(self, events: List[SecurityEvent]):
        """Analyze access patterns for suspicious activity."""
        try:
            # Group events by IP
            ip_events = {}
            for event in events:
                if event.ip_address not in ip_events:
                    ip_events[event.ip_address] = []
                ip_events[event.ip_address].append(event)
            
            # Analyze each IP's access pattern
            for ip, ip_event_list in ip_events.items():
                if len(ip_event_list) > 50:  # Suspicious number of requests
                    await self._create_threat(
                        ThreatType.DDoS_ATTEMPT,
                        ip,
                        None,
                        ip_event_list[0].user_agent,
                        {"request_count": len(ip_event_list), "window": 300},
                        SecurityLevel.MEDIUM
                    )
                
        except Exception as e:
            logger.error(f"Access pattern analysis failed: {e}")
    
    async def _check_threat_patterns(self):
        """Check for known threat patterns in recent events."""
        try:
            current_time = time.time()
            recent_events = [
                event for event in self.security_events
                if current_time - event.timestamp < 60  # Last minute
            ]
            
            for event in recent_events:
                # Check for injection attempts
                if "input" in event.details:
                    input_data = str(event.details["input"])
                    
                    for pattern_type, patterns in self.threat_patterns.items():
                        for pattern in patterns:
                            if pattern.search(input_data):
                                await self._create_threat(
                                    ThreatType.INJECTION_ATTEMPT,
                                    event.ip_address,
                                    event.user_id,
                                    event.user_agent,
                                    {
                                        "pattern_type": pattern_type,
                                        "matched_pattern": pattern.pattern,
                                        "input_data": input_data[:100]  # Truncate for security
                                    },
                                    SecurityLevel.HIGH
                                )
                                break
                
        except Exception as e:
            logger.error(f"Threat pattern checking failed: {e}")
    
    async def _create_threat(
        self,
        threat_type: ThreatType,
        ip_address: str,
        user_id: Optional[str],
        user_agent: str,
        details: Dict[str, Any],
        severity: SecurityLevel
    ):
        """Create a new threat record."""
        try:
            threat = Threat(
                id=secrets.token_urlsafe(16),
                threat_type=threat_type,
                timestamp=time.time(),
                ip_address=ip_address,
                user_id=user_id,
                user_agent=user_agent,
                details=details,
                severity=severity,
                blocked=False
            )
            
            self.threats.append(threat)
            
            # Log security event
            await self.log_security_event(
                SecurityEventType.THREAT_DETECTED,
                user_id,
                ip_address,
                user_agent,
                {"threat_id": threat.id, "threat_type": threat_type.value},
                severity
            )
            
            logger.warning(f"Threat detected: {threat_type.value} from {ip_address}")
            
        except Exception as e:
            logger.error(f"Failed to create threat record: {e}")
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        details: Dict[str, Any],
        severity: SecurityLevel,
        session_id: Optional[str] = None
    ):
        """Log a security event."""
        try:
            event = SecurityEvent(
                id=secrets.token_urlsafe(16),
                event_type=event_type,
                timestamp=time.time(),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                severity=severity,
                source="security_manager",
                session_id=session_id
            )
            
            self.security_events.append(event)
            
            # Keep only last 10000 events
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-10000:]
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked."""
        if ip_address in self.blocked_ips:
            if time.time() < self.blocked_ips[ip_address]:
                return True
            else:
                # Remove expired block
                del self.blocked_ips[ip_address]
        return False
    
    def is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if an IP address is suspicious."""
        try:
            # Check if it's a private/local IP
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private or ip.is_loopback:
                return False
            
            # Check if it's been involved in threats
            threat_count = sum(1 for threat in self.threats if threat.ip_address == ip_address)
            return threat_count > 3
            
        except Exception:
            return True
    
    async def validate_input(self, input_data: str, input_type: str = "general") -> Tuple[bool, Optional[str]]:
        """Validate input for security threats."""
        try:
            if not input_data:
                return True, None
            
            # Check for injection patterns
            for pattern_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if pattern.search(input_data):
                        await self._create_threat(
                            ThreatType.INJECTION_ATTEMPT,
                            "unknown",  # IP not available in this context
                            None,
                            "unknown",
                            {
                                "pattern_type": pattern_type,
                                "matched_pattern": pattern.pattern,
                                "input_type": input_type,
                                "input_data": input_data[:100]
                            },
                            SecurityLevel.HIGH
                        )
                        return False, f"Potential {pattern_type} detected"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, "Validation error"
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            if not self.enable_encryption:
                return data
            
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt encrypted data."""
        try:
            if not self.enable_encryption:
                return encrypted_data
            
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode(), salt)
            return hashed.decode()
            
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode(), hashed_password.encode())
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_jwt_token(self, user_id: str, permissions: List[str], expires_in: int = 3600) -> str:
        """Generate a JWT token."""
        try:
            payload = {
                "user_id": user_id,
                "permissions": permissions,
                "exp": datetime.utcnow() + timedelta(seconds=expires_in),
                "iat": datetime.utcnow(),
                "iss": "heygen_ai_security"
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            return token
            
        except Exception as e:
            logger.error(f"JWT token generation failed: {e}")
            raise
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT token verification failed: {e}")
            return None
    
    async def create_user_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        permissions: List[str],
        security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> str:
        """Create a new user session."""
        try:
            # Check concurrent session limit
            user_sessions = [
                session for session in self.user_sessions.values()
                if session.user_id == user_id and session.is_active
            ]
            
            if len(user_sessions) >= self.max_concurrent_sessions:
                # Deactivate oldest session
                oldest_session = min(user_sessions, key=lambda s: s.created_at)
                oldest_session.is_active = False
            
            # Create new session
            session_id = secrets.token_urlsafe(32)
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=time.time(),
                last_activity=time.time(),
                expires_at=time.time() + self.session_timeout,
                is_active=True,
                permissions=permissions,
                security_level=security_level
            )
            
            self.user_sessions[session_id] = session
            
            # Log security event
            await self.log_security_event(
                SecurityEventType.ACCESS_GRANTED,
                user_id,
                ip_address,
                user_agent,
                {"session_id": session_id, "permissions": permissions},
                SecurityLevel.LOW,
                session_id
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    def get_user_session(self, session_id: str) -> Optional[UserSession]:
        """Get a user session by ID."""
        session = self.user_sessions.get(session_id)
        
        if session and session.is_active and time.time() < session.expires_at:
            # Update last activity
            session.last_activity = time.time()
            return session
        
        return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a user session."""
        try:
            if session_id in self.user_sessions:
                session = self.user_sessions[session_id]
                session.is_active = False
                
                # Log security event
                await self.log_security_event(
                    SecurityEventType.LOGOUT,
                    session.user_id,
                    session.ip_address,
                    session.user_agent,
                    {"session_id": session_id},
                    SecurityLevel.LOW,
                    session_id
                )
                
                logger.info(f"Session {session_id} invalidated")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Session invalidation failed: {e}")
            return False
    
    def check_permission(self, session_id: str, required_permission: str) -> bool:
        """Check if a session has a specific permission."""
        session = self.get_user_session(session_id)
        if not session:
            return False
        
        return required_permission in session.permissions
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        current_time = time.time()
        
        return {
            "threats": {
                "total": len(self.threats),
                "active": len([t for t in self.threats if not t.resolved]),
                "blocked": len([t for t in self.threats if t.blocked]),
                "by_type": {
                    threat_type.value: len([t for t in self.threats if t.threat_type == threat_type])
                    for threat_type in ThreatType
                }
            },
            "blocked_ips": len([ip for ip, block_until in self.blocked_ips.items() if current_time < block_until]),
            "active_sessions": len([s for s in self.user_sessions.values() if s.is_active and current_time < s.expires_at]),
            "recent_events": len([e for e in self.security_events if current_time - e.timestamp < 3600]),
            "policies": {
                "total": len(self.security_policies),
                "enabled": len([p for p in self.security_policies if p.enabled])
            }
        }
    
    async def shutdown(self):
        """Shutdown the security manager."""
        # Cancel background tasks
        for task in [self.monitoring_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Advanced security manager shutdown complete")


# Global security manager instance
security_manager: Optional[AdvancedSecurityManager] = None

def get_security_manager() -> AdvancedSecurityManager:
    """Get global security manager instance."""
    global security_manager
    if security_manager is None:
        security_manager = AdvancedSecurityManager()
    return security_manager

async def shutdown_security_manager():
    """Shutdown global security manager."""
    global security_manager
    if security_manager:
        await security_manager.shutdown()
        security_manager = None

