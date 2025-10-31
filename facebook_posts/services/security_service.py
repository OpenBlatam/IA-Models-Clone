"""
Advanced Security Service for Facebook Posts API
Comprehensive security, authentication, authorization, and threat protection
"""

import asyncio
import hashlib
import hmac
import jwt
import secrets
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import cryptography
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from ..core.config import get_settings
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)


class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Threat type enumeration"""
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALICIOUS_CONTENT = "malicious_content"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    DDoS = "ddos"
    INJECTION = "injection"


class AuthenticationMethod(Enum):
    """Authentication method enumeration"""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SAML = "saml"
    LDAP = "ldap"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    id: str
    event_type: str
    threat_type: ThreatType
    security_level: SecurityLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class UserSession:
    """User session data structure"""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy data structure"""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enabled: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class PasswordManager:
    """Advanced password management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if not BCRYPT_AVAILABLE:
            # Fallback to hashlib if bcrypt not available
            salt = secrets.token_hex(16)
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if not BCRYPT_AVAILABLE:
            # Fallback verification
            try:
                salt = hashed_password[:32]  # Extract salt
                hash_part = hashed_password[32:]
                test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
                return hmac.compare_digest(test_hash, hash_part)
            except:
                return False
        
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate secure random password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        # Uppercase check
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Password should contain uppercase letters")
        
        # Lowercase check
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Password should contain lowercase letters")
        
        # Digit check
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Password should contain numbers")
        
        # Special character check
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Password should contain special characters")
        
        # Common password check
        common_passwords = ["password", "123456", "qwerty", "admin", "letmein"]
        if password.lower() in common_passwords:
            score = 0
            feedback.append("Password is too common")
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"]
        strength = strength_levels[min(score, len(strength_levels) - 1)]
        
        return {
            "score": score,
            "strength": strength,
            "feedback": feedback,
            "is_strong": score >= 4
        }


class TokenManager:
    """Advanced token management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = get_cache_manager()
        self.secret_key = self.settings.secret_key
        self.algorithm = self.settings.jwt_algorithm
        self.expiration = self.settings.jwt_expiration
    
    def generate_access_token(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate JWT access token"""
        payload = {
            "user_id": user_id,
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=self.expiration),
            "jti": secrets.token_urlsafe(32)
        }
        
        if metadata:
            payload.update(metadata)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def generate_refresh_token(self, user_id: str) -> str:
        """Generate refresh token"""
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=30),
            "jti": secrets.token_urlsafe(32)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke token by adding to blacklist"""
        try:
            payload = self.verify_token(token)
            if payload:
                jti = payload.get("jti")
                if jti:
                    # Add to blacklist cache
                    asyncio.create_task(self.cache_manager.cache.set(
                        f"blacklist:{jti}", "revoked", ttl=self.expiration
                    ))
                    return True
            return False
        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
            return False
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            payload = self.verify_token(token)
            if payload:
                jti = payload.get("jti")
                if jti:
                    # Check blacklist cache
                    blacklisted = asyncio.run(self.cache_manager.cache.get(f"blacklist:{jti}"))
                    return blacklisted is not None
            return False
        except Exception as e:
            logger.error("Failed to check token blacklist", error=str(e))
            return False


class EncryptionManager:
    """Advanced encryption management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = get_cache_manager()
        self._fernet = None
        self._initialize_fernet()
    
    def _initialize_fernet(self):
        """Initialize Fernet encryption"""
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available - encryption disabled")
            return
        
        try:
            # Get or generate encryption key
            encryption_key = self.settings.encryption_key
            if not encryption_key:
                encryption_key = Fernet.generate_key()
                logger.warning("Generated new encryption key - store it securely")
            
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()
            
            self._fernet = Fernet(encryption_key)
            logger.info("Encryption manager initialized")
        except Exception as e:
            logger.error("Failed to initialize encryption", error=str(e))
    
    def encrypt_data(self, data: str) -> Optional[str]:
        """Encrypt sensitive data"""
        if not self._fernet:
            logger.warning("Encryption not available")
            return data
        
        try:
            encrypted_data = self._fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error("Failed to encrypt data", error=str(e))
            return None
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data"""
        if not self._fernet:
            logger.warning("Encryption not available")
            return encrypted_data
        
        try:
            decrypted_data = self._fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Failed to decrypt data", error=str(e))
            return None
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage"""
        salt = secrets.token_hex(16)
        hash_value = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return f"{salt}:{hash_value.hex()}"
    
    def verify_hashed_data(self, data: str, hashed_data: str) -> bool:
        """Verify hashed sensitive data"""
        try:
            salt, hash_part = hashed_data.split(':')
            test_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
            return hmac.compare_digest(test_hash.hex(), hash_part)
        except:
            return False


class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.failed_attempts = {}
        self.suspicious_ips = set()
        self.rate_limits = {}
    
    @timed("threat_detection")
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect security threats in request"""
        threats = []
        
        # Extract request information
        ip_address = request_data.get("ip_address", "unknown")
        user_agent = request_data.get("user_agent", "unknown")
        user_id = request_data.get("user_id")
        endpoint = request_data.get("endpoint", "unknown")
        
        # Check for brute force attacks
        brute_force_threat = await self._detect_brute_force(ip_address, user_id)
        if brute_force_threat:
            threats.append(brute_force_threat)
        
        # Check for rate limiting violations
        rate_limit_threat = await self._detect_rate_limit_violation(ip_address, endpoint)
        if rate_limit_threat:
            threats.append(rate_limit_threat)
        
        # Check for suspicious activity
        suspicious_threat = await self._detect_suspicious_activity(request_data)
        if suspicious_threat:
            threats.append(suspicious_threat)
        
        # Check for malicious content
        malicious_threat = await self._detect_malicious_content(request_data)
        if malicious_threat:
            threats.append(malicious_threat)
        
        return threats
    
    async def _detect_brute_force(self, ip_address: str, user_id: Optional[str]) -> Optional[SecurityEvent]:
        """Detect brute force attacks"""
        try:
            # Check failed login attempts
            cache_key = f"failed_attempts:{ip_address}"
            failed_attempts = await self.cache_manager.cache.get(cache_key) or 0
            
            if failed_attempts >= 5:  # Threshold for brute force
                return SecurityEvent(
                    id=f"brute_force_{int(time.time())}",
                    event_type="authentication_failure",
                    threat_type=ThreatType.BRUTE_FORCE,
                    security_level=SecurityLevel.HIGH,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent="unknown",
                    description=f"Brute force attack detected from {ip_address}",
                    metadata={"failed_attempts": failed_attempts}
                )
            
            return None
        except Exception as e:
            logger.error("Failed to detect brute force", error=str(e))
            return None
    
    async def _detect_rate_limit_violation(self, ip_address: str, endpoint: str) -> Optional[SecurityEvent]:
        """Detect rate limit violations"""
        try:
            # Check rate limiting
            cache_key = f"rate_limit:{ip_address}:{endpoint}"
            request_count = await self.cache_manager.cache.get(cache_key) or 0
            
            if request_count > 100:  # Threshold for rate limiting
                return SecurityEvent(
                    id=f"rate_limit_{int(time.time())}",
                    event_type="rate_limit_exceeded",
                    threat_type=ThreatType.RATE_LIMIT_EXCEEDED,
                    security_level=SecurityLevel.MEDIUM,
                    user_id=None,
                    ip_address=ip_address,
                    user_agent="unknown",
                    description=f"Rate limit exceeded from {ip_address} on {endpoint}",
                    metadata={"request_count": request_count, "endpoint": endpoint}
                )
            
            return None
        except Exception as e:
            logger.error("Failed to detect rate limit violation", error=str(e))
            return None
    
    async def _detect_suspicious_activity(self, request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect suspicious activity patterns"""
        try:
            ip_address = request_data.get("ip_address", "unknown")
            user_agent = request_data.get("user_agent", "unknown")
            endpoint = request_data.get("endpoint", "unknown")
            
            # Check for suspicious user agents
            suspicious_agents = ["bot", "crawler", "spider", "scraper"]
            if any(agent in user_agent.lower() for agent in suspicious_agents):
                return SecurityEvent(
                    id=f"suspicious_{int(time.time())}",
                    event_type="suspicious_activity",
                    threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                    security_level=SecurityLevel.MEDIUM,
                    user_id=None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    description=f"Suspicious user agent detected: {user_agent}",
                    metadata={"endpoint": endpoint}
                )
            
            # Check for rapid endpoint switching
            cache_key = f"endpoint_switching:{ip_address}"
            endpoints = await self.cache_manager.cache.get(cache_key) or []
            endpoints.append(endpoint)
            
            if len(set(endpoints)) > 10:  # More than 10 different endpoints
                return SecurityEvent(
                    id=f"endpoint_switching_{int(time.time())}",
                    event_type="suspicious_activity",
                    threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                    security_level=SecurityLevel.MEDIUM,
                    user_id=None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    description=f"Rapid endpoint switching detected from {ip_address}",
                    metadata={"endpoints": list(set(endpoints))}
                )
            
            # Update cache
            await self.cache_manager.cache.set(cache_key, endpoints[-20:], ttl=3600)
            
            return None
        except Exception as e:
            logger.error("Failed to detect suspicious activity", error=str(e))
            return None
    
    async def _detect_malicious_content(self, request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect malicious content in requests"""
        try:
            # Check for SQL injection patterns
            sql_patterns = ["'", "union", "select", "drop", "insert", "update", "delete"]
            content = str(request_data.get("content", "")).lower()
            
            if any(pattern in content for pattern in sql_patterns):
                return SecurityEvent(
                    id=f"sql_injection_{int(time.time())}",
                    event_type="malicious_content",
                    threat_type=ThreatType.INJECTION,
                    security_level=SecurityLevel.HIGH,
                    user_id=request_data.get("user_id"),
                    ip_address=request_data.get("ip_address", "unknown"),
                    user_agent=request_data.get("user_agent", "unknown"),
                    description="Potential SQL injection attempt detected",
                    metadata={"content": content[:100]}  # Truncate for security
                )
            
            # Check for XSS patterns
            xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
            if any(pattern in content for pattern in xss_patterns):
                return SecurityEvent(
                    id=f"xss_{int(time.time())}",
                    event_type="malicious_content",
                    threat_type=ThreatType.INJECTION,
                    security_level=SecurityLevel.HIGH,
                    user_id=request_data.get("user_id"),
                    ip_address=request_data.get("ip_address", "unknown"),
                    user_agent=request_data.get("user_agent", "unknown"),
                    description="Potential XSS attempt detected",
                    metadata={"content": content[:100]}  # Truncate for security
                )
            
            return None
        except Exception as e:
            logger.error("Failed to detect malicious content", error=str(e))
            return None
    
    async def record_failed_attempt(self, ip_address: str, user_id: Optional[str] = None):
        """Record failed authentication attempt"""
        try:
            cache_key = f"failed_attempts:{ip_address}"
            failed_attempts = await self.cache_manager.cache.get(cache_key) or 0
            failed_attempts += 1
            
            await self.cache_manager.cache.set(cache_key, failed_attempts, ttl=3600)
            
            logger.warning("Failed authentication attempt recorded", 
                         ip_address=ip_address, user_id=user_id, attempts=failed_attempts)
        except Exception as e:
            logger.error("Failed to record failed attempt", error=str(e))
    
    async def record_successful_attempt(self, ip_address: str, user_id: Optional[str] = None):
        """Record successful authentication attempt"""
        try:
            cache_key = f"failed_attempts:{ip_address}"
            await self.cache_manager.cache.delete(cache_key)
            
            logger.info("Successful authentication attempt recorded", 
                       ip_address=ip_address, user_id=user_id)
        except Exception as e:
            logger.error("Failed to record successful attempt", error=str(e))


class SecurityPolicyEngine:
    """Security policy enforcement engine"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.cache_manager = get_cache_manager()
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default security policies"""
        default_policies = [
            SecurityPolicy(
                id="password_policy",
                name="Password Policy",
                description="Enforce strong password requirements",
                rules=[
                    {"type": "min_length", "value": 8},
                    {"type": "require_uppercase", "value": True},
                    {"type": "require_lowercase", "value": True},
                    {"type": "require_numbers", "value": True},
                    {"type": "require_special_chars", "value": True}
                ],
                priority=1
            ),
            SecurityPolicy(
                id="session_policy",
                name="Session Policy",
                description="Enforce session security requirements",
                rules=[
                    {"type": "max_session_duration", "value": 3600},  # 1 hour
                    {"type": "require_https", "value": True},
                    {"type": "secure_cookies", "value": True}
                ],
                priority=2
            ),
            SecurityPolicy(
                id="rate_limit_policy",
                name="Rate Limiting Policy",
                description="Enforce rate limiting rules",
                rules=[
                    {"type": "max_requests_per_minute", "value": 60},
                    {"type": "max_requests_per_hour", "value": 1000},
                    {"type": "max_requests_per_day", "value": 10000}
                ],
                priority=3
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.id] = policy
    
    async def evaluate_policy(self, policy_id: str, context: Dict[str, Any]) -> bool:
        """Evaluate security policy against context"""
        policy = self.policies.get(policy_id)
        if not policy or not policy.enabled:
            return True
        
        try:
            for rule in policy.rules:
                if not self._evaluate_rule(rule, context):
                    return False
            return True
        except Exception as e:
            logger.error("Failed to evaluate policy", policy_id=policy_id, error=str(e))
            return False
    
    def _evaluate_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate individual policy rule"""
        rule_type = rule.get("type")
        rule_value = rule.get("value")
        
        if rule_type == "min_length":
            password = context.get("password", "")
            return len(password) >= rule_value
        elif rule_type == "require_uppercase":
            password = context.get("password", "")
            return any(c.isupper() for c in password) if rule_value else True
        elif rule_type == "require_lowercase":
            password = context.get("password", "")
            return any(c.islower() for c in password) if rule_value else True
        elif rule_type == "require_numbers":
            password = context.get("password", "")
            return any(c.isdigit() for c in password) if rule_value else True
        elif rule_type == "require_special_chars":
            password = context.get("password", "")
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            return any(c in special_chars for c in password) if rule_value else True
        elif rule_type == "max_session_duration":
            session_duration = context.get("session_duration", 0)
            return session_duration <= rule_value
        elif rule_type == "require_https":
            is_https = context.get("is_https", False)
            return is_https if rule_value else True
        elif rule_type == "secure_cookies":
            secure_cookies = context.get("secure_cookies", False)
            return secure_cookies if rule_value else True
        elif rule_type == "max_requests_per_minute":
            requests_per_minute = context.get("requests_per_minute", 0)
            return requests_per_minute <= rule_value
        elif rule_type == "max_requests_per_hour":
            requests_per_hour = context.get("requests_per_hour", 0)
            return requests_per_hour <= rule_value
        elif rule_type == "max_requests_per_day":
            requests_per_day = context.get("requests_per_day", 0)
            return requests_per_day <= rule_value
        else:
            return True


class SecurityService:
    """Main security service orchestrator"""
    
    def __init__(self):
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager()
        self.encryption_manager = EncryptionManager()
        self.threat_detector = ThreatDetector()
        self.policy_engine = SecurityPolicyEngine()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.active_sessions: Dict[str, UserSession] = {}
    
    @timed("security_authentication")
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user with security checks"""
        try:
            # Detect threats
            request_data = {
                "ip_address": ip_address,
                "user_agent": user_agent,
                "username": username
            }
            
            threats = await self.threat_detector.detect_threats(request_data)
            if threats:
                # Log threats
                for threat in threats:
                    logger.warning("Security threat detected during authentication", 
                                 threat_id=threat.id, threat_type=threat.threat_type.value)
                
                # Block authentication if high-level threat
                high_level_threats = [t for t in threats if t.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]]
                if high_level_threats:
                    await self.threat_detector.record_failed_attempt(ip_address, username)
                    return None
            
            # Mock authentication (in real implementation, verify against database)
            if username == "admin" and password == "admin123":
                # Generate tokens
                access_token = self.token_manager.generate_access_token(username)
                refresh_token = self.token_manager.generate_refresh_token(username)
                
                # Create session
                session = UserSession(
                    session_id=secrets.token_urlsafe(32),
                    user_id=username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=1)
                )
                
                self.active_sessions[session.session_id] = session
                
                # Record successful attempt
                await self.threat_detector.record_successful_attempt(ip_address, username)
                
                return {
                    "user_id": username,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "session_id": session.session_id,
                    "expires_at": session.expires_at.isoformat()
                }
            else:
                # Record failed attempt
                await self.threat_detector.record_failed_attempt(ip_address, username)
                return None
                
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            return None
    
    @timed("security_authorization")
    async def authorize_request(
        self,
        token: str,
        required_permissions: List[str],
        ip_address: str,
        user_agent: str
    ) -> Optional[Dict[str, Any]]:
        """Authorize request with security checks"""
        try:
            # Verify token
            payload = self.token_manager.verify_token(token)
            if not payload:
                return None
            
            # Check if token is blacklisted
            if self.token_manager.is_token_blacklisted(token):
                return None
            
            # Detect threats
            request_data = {
                "ip_address": ip_address,
                "user_agent": user_agent,
                "user_id": payload.get("user_id")
            }
            
            threats = await self.threat_detector.detect_threats(request_data)
            if threats:
                # Log threats
                for threat in threats:
                    logger.warning("Security threat detected during authorization", 
                                 threat_id=threat.id, threat_type=threat.threat_type.value)
                
                # Block authorization if high-level threat
                high_level_threats = [t for t in threats if t.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]]
                if high_level_threats:
                    return None
            
            # Mock authorization (in real implementation, check user permissions)
            user_id = payload.get("user_id")
            user_permissions = ["read", "write", "admin"]  # Mock permissions
            
            has_permission = any(perm in user_permissions for perm in required_permissions)
            if not has_permission:
                return None
            
            return {
                "user_id": user_id,
                "permissions": user_permissions,
                "token_payload": payload
            }
            
        except Exception as e:
            logger.error("Authorization failed", error=str(e))
            return None
    
    @timed("security_encryption")
    async def encrypt_sensitive_data(self, data: str) -> Optional[str]:
        """Encrypt sensitive data"""
        return self.encryption_manager.encrypt_data(data)
    
    @timed("security_decryption")
    async def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data"""
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    @timed("security_password_validation")
    async def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security policies"""
        context = {"password": password}
        
        # Check password policy
        policy_compliant = await self.policy_engine.evaluate_policy("password_policy", context)
        
        # Get password strength
        strength_analysis = self.password_manager.validate_password_strength(password)
        
        return {
            "is_valid": policy_compliant and strength_analysis["is_strong"],
            "policy_compliant": policy_compliant,
            "strength_analysis": strength_analysis
        }
    
    @timed("security_session_management")
    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str
    ) -> UserSession:
        """Create secure user session"""
        session = UserSession(
            session_id=secrets.token_urlsafe(32),
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        self.active_sessions[session.session_id] = session
        return session
    
    @timed("security_session_validation")
    async def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate user session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if datetime.now() > session.expires_at:
            del self.active_sessions[session_id]
            return None
        
        # Update last activity
        session.last_activity = datetime.now()
        return session
    
    @timed("security_logout")
    async def logout_user(self, session_id: str, token: Optional[str] = None):
        """Logout user and invalidate session"""
        try:
            # Remove session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Revoke token if provided
            if token:
                self.token_manager.revoke_token(token)
            
            logger.info("User logged out successfully", session_id=session_id)
        except Exception as e:
            logger.error("Logout failed", session_id=session_id, error=str(e))


# Global security service instance
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """Get global security service instance"""
    global _security_service
    
    if _security_service is None:
        _security_service = SecurityService()
    
    return _security_service


# Export all classes and functions
__all__ = [
    # Enums
    'SecurityLevel',
    'ThreatType',
    'AuthenticationMethod',
    
    # Data classes
    'SecurityEvent',
    'UserSession',
    'SecurityPolicy',
    
    # Services
    'PasswordManager',
    'TokenManager',
    'EncryptionManager',
    'ThreatDetector',
    'SecurityPolicyEngine',
    'SecurityService',
    
    # Utility functions
    'get_security_service',
]






























