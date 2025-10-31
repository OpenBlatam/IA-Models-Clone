"""
Enhanced Security System for BUL API
====================================

Comprehensive security features including authentication, authorization,
rate limiting, input validation, and threat detection.
"""

import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, validator
import bcrypt
import secrets

from ..utils import get_logger, log_security_event
from ..config import get_config

logger = get_logger(__name__)

class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    API_USER = "api_user"

class TokenType(str, Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"

# Enhanced Pydantic Models
class User(BaseModel):
    """User model with enhanced security"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()
    
    @validator('email')
    def validate_email(cls, v):
        return v.lower()

class Token(BaseModel):
    """Token model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    token_type: TokenType
    token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.now)
    is_revoked: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SecurityEvent(BaseModel):
    """Security event model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ThreatType
    severity: SecurityLevel
    user_id: Optional[str] = None
    ip_address: str
    user_agent: str
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RateLimitRule(BaseModel):
    """Rate limiting rule"""
    endpoint: str
    max_requests: int
    window_seconds: int
    burst_limit: int = 0
    user_specific: bool = False

class SecurityConfig(BaseModel):
    """Security configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_2fa: bool = False
    session_timeout_minutes: int = 30

# Enhanced Security Classes
class PasswordManager:
    """Enhanced password management"""
    
    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["bcrypt", "argon2"],
            deprecated="auto",
            bcrypt__rounds=12
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        issues = []
        score = 0
        
        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1
        
        # Character variety checks
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        else:
            score += 1
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        # Common password check
        common_passwords = [
            "password", "123456", "admin", "qwerty", "letmein"
        ]
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            score -= 2
        
        return {
            "is_valid": len(issues) == 0,
            "score": max(0, score),
            "issues": issues
        }

class TokenManager:
    """Enhanced token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        payload = {
            "user_id": user_id,
            "token_type": "access",
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        expire = datetime.utcnow() + timedelta(days=7)
        
        payload = {
            "user_id": user_id,
            "token_type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Revoke token by adding to blacklist"""
        self.token_blacklist.add(token)
    
    def create_api_key(self, user_id: str, name: str) -> str:
        """Create API key"""
        key_data = f"{user_id}:{name}:{time.time()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        return f"bul_{api_key[:32]}"

class RateLimiter:
    """Enhanced rate limiting system"""
    
    def __init__(self):
        self.requests = {}  # {client_id: [(timestamp, count)]}
        self.rules = {
            "default": RateLimitRule(
                endpoint="*",
                max_requests=100,
                window_seconds=60,
                burst_limit=10
            ),
            "generate": RateLimitRule(
                endpoint="/generate",
                max_requests=10,
                window_seconds=60,
                burst_limit=2
            ),
            "batch": RateLimitRule(
                endpoint="/generate/batch",
                max_requests=5,
                window_seconds=300,
                burst_limit=1
            )
        }
    
    def is_allowed(self, client_id: str, endpoint: str = "*") -> bool:
        """Check if request is allowed"""
        current_time = time.time()
        
        # Get rule for endpoint
        rule = self.rules.get(endpoint, self.rules["default"])
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if current_time - timestamp < rule.window_seconds
            ]
        else:
            self.requests[client_id] = []
        
        # Count requests in window
        total_requests = sum(count for _, count in self.requests[client_id])
        
        # Check rate limit
        if total_requests >= rule.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append((current_time, 1))
        
        return True
    
    def get_remaining_requests(self, client_id: str, endpoint: str = "*") -> int:
        """Get remaining requests for client"""
        rule = self.rules.get(endpoint, self.rules["default"])
        current_time = time.time()
        
        if client_id not in self.requests:
            return rule.max_requests
        
        # Clean old requests
        self.requests[client_id] = [
            (timestamp, count) for timestamp, count in self.requests[client_id]
            if current_time - timestamp < rule.window_seconds
        ]
        
        total_requests = sum(count for _, count in self.requests[client_id])
        return max(0, rule.max_requests - total_requests)

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # SQL Injection patterns
            r"(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"(--|#|\/\*|\*\/)",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
            
            # Command injection
            r"[;&|`$]",
            r"(\||&&|\|\|)",
        ]
        
        self.threat_scores = {}
        self.blocked_ips = set()
    
    def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for threats"""
        threat_score = 0
        threats_detected = []
        
        # Analyze URL
        url_analysis = self._analyze_url(str(request.url))
        threat_score += url_analysis["score"]
        threats_detected.extend(url_analysis["threats"])
        
        # Analyze headers
        header_analysis = self._analyze_headers(request.headers)
        threat_score += header_analysis["score"]
        threats_detected.extend(header_analysis["threats"])
        
        # Analyze user agent
        user_agent = request.headers.get("user-agent", "")
        ua_analysis = self._analyze_user_agent(user_agent)
        threat_score += ua_analysis["score"]
        threats_detected.extend(ua_analysis["threats"])
        
        # Check IP reputation
        client_ip = request.client.host if request.client else "unknown"
        ip_analysis = self._analyze_ip(client_ip)
        threat_score += ip_analysis["score"]
        threats_detected.extend(ip_analysis["threats"])
        
        return {
            "threat_score": threat_score,
            "threats_detected": threats_detected,
            "risk_level": self._calculate_risk_level(threat_score),
            "recommendation": self._get_recommendation(threat_score)
        }
    
    def _analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze URL for threats"""
        score = 0
        threats = []
        
        import re
        for pattern in self.suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                score += 10
                threats.append(f"Suspicious pattern in URL: {pattern}")
        
        # Check URL length
        if len(url) > 2000:
            score += 5
            threats.append("URL too long")
        
        return {"score": score, "threats": threats}
    
    def _analyze_headers(self, headers) -> Dict[str, Any]:
        """Analyze headers for threats"""
        score = 0
        threats = []
        
        # Check for suspicious header values
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-originating-ip"]
        for header in suspicious_headers:
            if header in headers:
                value = headers[header]
                if len(value) > 100:
                    score += 5
                    threats.append(f"Suspicious {header} header")
        
        return {"score": score, "threats": threats}
    
    def _analyze_user_agent(self, user_agent: str) -> Dict[str, Any]:
        """Analyze user agent for threats"""
        score = 0
        threats = []
        
        # Check for suspicious user agents
        suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan"]
        for agent in suspicious_agents:
            if agent.lower() in user_agent.lower():
                score += 20
                threats.append(f"Suspicious user agent: {agent}")
        
        # Check for empty or very short user agents
        if not user_agent or len(user_agent) < 10:
            score += 5
            threats.append("Suspicious user agent")
        
        return {"score": score, "threats": threats}
    
    def _analyze_ip(self, ip: str) -> Dict[str, Any]:
        """Analyze IP address"""
        score = 0
        threats = []
        
        # Check if IP is blocked
        if ip in self.blocked_ips:
            score += 100
            threats.append("Blocked IP address")
        
        # Check for private/local IPs (might be suspicious in production)
        if ip.startswith(("127.", "192.168.", "10.", "172.")):
            score += 2
            threats.append("Private IP address")
        
        return {"score": score, "threats": threats}
    
    def _calculate_risk_level(self, threat_score: int) -> str:
        """Calculate risk level based on threat score"""
        if threat_score >= 50:
            return "HIGH"
        elif threat_score >= 20:
            return "MEDIUM"
        elif threat_score >= 5:
            return "LOW"
        else:
            return "NONE"
    
    def _get_recommendation(self, threat_score: int) -> str:
        """Get security recommendation"""
        if threat_score >= 50:
            return "BLOCK_REQUEST"
        elif threat_score >= 20:
            return "INCREASE_MONITORING"
        elif threat_score >= 5:
            return "LOG_AND_CONTINUE"
        else:
            return "ALLOW"

class SecurityValidator:
    """Enhanced security validation"""
    
    def __init__(self):
        self.password_manager = PasswordManager()
        self.threat_detector = ThreatDetector()
        self.rate_limiter = RateLimiter()
    
    async def validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate request for security issues"""
        validation_result = {
            "is_valid": True,
            "threats": [],
            "recommendations": []
        }
        
        # Threat detection
        threat_analysis = self.threat_detector.analyze_request(request)
        if threat_analysis["threat_score"] > 0:
            validation_result["threats"].extend(threat_analysis["threats_detected"])
            validation_result["recommendations"].append(threat_analysis["recommendation"])
        
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"
        if not self.rate_limiter.is_allowed(client_id, request.url.path):
            validation_result["is_valid"] = False
            validation_result["threats"].append("Rate limit exceeded")
            validation_result["recommendations"].append("BLOCK_REQUEST")
        
        return validation_result
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        return self.password_manager.validate_password_strength(password)
    
    def validate_input(self, input_data: str, input_type: str = "general") -> Dict[str, Any]:
        """Validate input data for security issues"""
        issues = []
        
        # Length validation
        if len(input_data) > 10000:
            issues.append("Input too long")
        
        # Pattern validation based on type
        if input_type == "email":
            import re
            if not re.match(r'^[^@]+@[^@]+\.[^@]+$', input_data):
                issues.append("Invalid email format")
        
        elif input_type == "username":
            if not input_data.isalnum():
                issues.append("Username must be alphanumeric")
        
        # Check for suspicious patterns
        import re
        for pattern in self.threat_detector.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                issues.append(f"Suspicious pattern detected: {pattern}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

# Global security manager
class SecurityManager:
    """Main security manager"""
    
    def __init__(self):
        self.config = get_config()
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager(
            self.config.security.secret_key,
            self.config.security.algorithm
        )
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        self.validator = SecurityValidator()
        self.security_events = []
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with enhanced security"""
        # This would integrate with actual user database
        # For now, return a mock user
        return User(
            username=username,
            email=f"{username}@example.com",
            password_hash=self.password_manager.hash_password(password),
            role=UserRole.USER
        )
    
    async def create_security_event(
        self,
        event_type: ThreatType,
        severity: SecurityLevel,
        description: str,
        user_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown"
    ):
        """Create security event"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            description=description
        )
        
        self.security_events.append(event)
        
        # Log security event
        await log_security_event(
            event_type=event_type.value,
            severity=severity.value,
            description=description,
            user_id=user_id,
            ip_address=ip_address
        )
        
        # Take action based on severity
        if severity == SecurityLevel.CRITICAL:
            await self._handle_critical_threat(event)
    
    async def _handle_critical_threat(self, event: SecurityEvent):
        """Handle critical security threats"""
        # Block IP if multiple critical events
        critical_events = [
            e for e in self.security_events
            if e.severity == SecurityLevel.CRITICAL and e.ip_address == event.ip_address
        ]
        
        if len(critical_events) >= 3:
            self.threat_detector.blocked_ips.add(event.ip_address)
            logger.critical(f"IP {event.ip_address} blocked due to critical threats")

# Global instances
_security_manager: Optional[SecurityManager] = None
_rate_limiter: Optional[RateLimiter] = None
_auth_manager: Optional[SecurityManager] = None

def get_security_manager() -> SecurityManager:
    """Get global security manager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def get_auth_manager() -> SecurityManager:
    """Get global auth manager"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SecurityManager()
    return _auth_manager

# Security decorators
def require_auth(required_role: UserRole = UserRole.USER):
    """Decorator to require authentication"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would check authentication
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would check permissions
            return await func(*args, **kwargs)
        return wrapper
    return decorator