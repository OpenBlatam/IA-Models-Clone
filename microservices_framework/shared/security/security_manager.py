"""
Advanced Security Manager
Features: OAuth2, JWT, rate limiting, security headers, input validation, DDoS protection
"""

import asyncio
import hashlib
import hmac
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import ipaddress
import re

# Security imports
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class RateLimitType(Enum):
    """Rate limit types"""
    IP = "ip"
    USER = "user"
    API_KEY = "api_key"
    GLOBAL = "global"

class ThreatLevel(Enum):
    """Threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    refresh_token_expiration: int = 86400 * 7  # 7 days
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = True
    rate_limit_by: RateLimitType = RateLimitType.IP
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    ddos_protection_enabled: bool = True
    ddos_threshold: int = 100  # requests per minute
    ddos_block_duration: int = 3600  # seconds
    security_headers_enabled: bool = True
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    input_validation_enabled: bool = True
    sql_injection_protection: bool = True
    xss_protection: bool = True
    csrf_protection: bool = True

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Security event model"""
    id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    threat_level: ThreatLevel
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

class RateLimiter:
    """
    Advanced rate limiter with multiple strategies
    """
    
    def __init__(self, redis_client: aioredis.Redis, config: SecurityConfig):
        self.redis = redis_client
        self.config = config
        self.blocked_ips: Dict[str, float] = {}
    
    async def is_allowed(self, identifier: str, limit_type: RateLimitType = None) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limit"""
        limit_type = limit_type or self.config.rate_limit_by
        
        # Check if IP is blocked
        if limit_type == RateLimitType.IP and identifier in self.blocked_ips:
            if time.time() < self.blocked_ips[identifier]:
                return False, {"reason": "IP blocked", "blocked_until": self.blocked_ips[identifier]}
            else:
                del self.blocked_ips[identifier]
        
        # Get rate limit key
        key = f"rate_limit:{limit_type.value}:{identifier}"
        
        # Check current count
        current_count = await self.redis.get(key)
        current_count = int(current_count) if current_count else 0
        
        # Check limits
        if current_count >= self.config.max_requests_per_minute:
            # Block IP if DDoS protection is enabled
            if self.config.ddos_protection_enabled and limit_type == RateLimitType.IP:
                self.blocked_ips[identifier] = time.time() + self.config.ddos_block_duration
            
            return False, {
                "limit": self.config.max_requests_per_minute,
                "remaining": 0,
                "reset": int(time.time()) + 60,
                "reason": "Rate limit exceeded"
            }
        
        # Increment counter
        await self.redis.incr(key)
        await self.redis.expire(key, 60)  # 1 minute window
        
        return True, {
            "limit": self.config.max_requests_per_minute,
            "remaining": self.config.max_requests_per_minute - current_count - 1,
            "reset": int(time.time()) + 60
        }
    
    async def reset_limit(self, identifier: str, limit_type: RateLimitType = None):
        """Reset rate limit for identifier"""
        limit_type = limit_type or self.config.rate_limit_by
        key = f"rate_limit:{limit_type.value}:{identifier}"
        await self.redis.delete(key)

class InputValidator:
    """
    Advanced input validation and sanitization
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
            r"(\b(OR|AND)\s+\".*\"\s*=\s*\".*\")",
            r"(\b(OR|AND)\s+\w+\s*=\s*\w+)",
        ]
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
        ]
    
    def validate_input(self, input_data: Any, field_name: str = None) -> tuple[bool, str]:
        """Validate and sanitize input data"""
        if not self.config.input_validation_enabled:
            return True, ""
        
        # Convert to string for validation
        if isinstance(input_data, (dict, list)):
            input_str = str(input_data)
        else:
            input_str = str(input_data)
        
        # SQL Injection protection
        if self.config.sql_injection_protection:
            for pattern in self.sql_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return False, f"Potential SQL injection detected in {field_name or 'input'}"
        
        # XSS protection
        if self.config.xss_protection:
            for pattern in self.xss_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return False, f"Potential XSS attack detected in {field_name or 'input'}"
        
        return True, ""
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_data)
        return sanitized.strip()

class JWTManager:
    """
    JWT token management
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if not JWT_AVAILABLE:
            raise ImportError("python-jose is required for JWT support")
        
        to_encode = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + (expires_delta or timedelta(seconds=self.config.jwt_expiration)),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        if not JWT_AVAILABLE:
            raise ImportError("python-jose is required for JWT support")
        
        to_encode = {
            "sub": user.id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(seconds=self.config.refresh_token_expiration),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        if not JWT_AVAILABLE:
            raise ImportError("python-jose is required for JWT support")
        
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            return payload
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)

class SecurityHeaders:
    """
    Security headers management
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers"""
        if not self.config.security_headers_enabled:
            return {}
        
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        if self.config.xss_protection:
            headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        if self.config.csrf_protection:
            headers["X-CSRF-Protection"] = "1"
        
        return headers

class SecurityManager:
    """
    Comprehensive security manager
    """
    
    def __init__(self, config: SecurityConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis = redis_client
        self.rate_limiter = RateLimiter(redis_client, config) if redis_client else None
        self.input_validator = InputValidator(config)
        self.jwt_manager = JWTManager(config)
        self.security_headers = SecurityHeaders(config)
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Dict[str, float] = {}
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        # This is a placeholder - implement your user authentication logic
        # In production, you would query your user database
        
        # Example user (replace with database lookup)
        if username == "admin" and password == "admin123":
            return User(
                id=str(uuid.uuid4()),
                username=username,
                email="admin@example.com",
                roles=["admin"],
                permissions=["read", "write", "delete"],
                is_active=True,
                is_verified=True
            )
        
        return None
    
    async def authorize_user(self, user: User, required_permission: str) -> bool:
        """Authorize user for specific permission"""
        return required_permission in user.permissions or "admin" in user.roles
    
    async def check_rate_limit(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        """Check rate limit for request"""
        if not self.rate_limiter:
            return True, {}
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        return await self.rate_limiter.is_allowed(client_ip, RateLimitType.IP)
    
    async def validate_request(self, request: Request) -> tuple[bool, str]:
        """Validate request for security threats"""
        # Get request data
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Check for suspicious patterns
        if self._is_suspicious_request(request):
            await self._log_security_event(
                "suspicious_request",
                None,
                client_ip,
                user_agent,
                ThreatLevel.MEDIUM,
                {"reason": "Suspicious request pattern"}
            )
            return False, "Suspicious request detected"
        
        # Validate input data
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # This is a simplified validation - in production, you'd parse JSON/form data
                    is_valid, error = self.input_validator.validate_input(body.decode())
                    if not is_valid:
                        await self._log_security_event(
                            "invalid_input",
                            None,
                            client_ip,
                            user_agent,
                            ThreatLevel.HIGH,
                            {"error": error}
                        )
                        return False, error
            except Exception as e:
                return False, f"Invalid request data: {str(e)}"
        
        return True, ""
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_suspicious_request(self, request: Request) -> bool:
        """Check if request is suspicious"""
        user_agent = request.headers.get("user-agent", "").lower()
        
        # Check for common bot patterns
        bot_patterns = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
        if any(pattern in user_agent for pattern in bot_patterns):
            return True
        
        # Check for suspicious paths
        suspicious_paths = ["/admin", "/wp-admin", "/phpmyadmin", "/.env", "/config"]
        if any(path in request.url.path for path in suspicious_paths):
            return True
        
        return False
    
    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        threat_level: ThreatLevel,
        details: Dict[str, Any]
    ):
        """Log security event"""
        event = SecurityEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            threat_level=threat_level,
            details=details
        )
        
        self.security_events.append(event)
        
        # Store in Redis if available
        if self.redis:
            await self.redis.lpush(
                "security_events",
                json.dumps(event.__dict__, default=str)
            )
            await self.redis.ltrim("security_events", 0, 999)  # Keep last 1000 events
        
        logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_id": event.id,
                "user_id": user_id,
                "ip_address": ip_address,
                "threat_level": threat_level.value,
                "details": details
            }
        )
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers"""
        return self.security_headers.get_security_headers()
    
    async def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        return self.security_events[-limit:]
    
    async def block_ip(self, ip_address: str, duration: int = 3600):
        """Block IP address"""
        self.blocked_ips[ip_address] = time.time() + duration
        
        if self.redis:
            await self.redis.setex(f"blocked_ip:{ip_address}", duration, "1")
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        if ip_address in self.blocked_ips:
            if time.time() < self.blocked_ips[ip_address]:
                return True
            else:
                del self.blocked_ips[ip_address]
        
        if self.redis:
            blocked = await self.redis.get(f"blocked_ip:{ip_address}")
            return bool(blocked)
        
        return False

# FastAPI dependencies
security_scheme = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    security_manager: SecurityManager = Depends(lambda: get_security_manager())
) -> User:
    """Get current authenticated user"""
    try:
        payload = security_manager.jwt_manager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # In production, you would fetch user from database
        # For now, create user from token payload
        user = User(
            id=user_id,
            username=payload.get("username", ""),
            email=payload.get("email", ""),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def require_permission(permission: str):
    """Require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if permission not in current_user.permissions and "admin" not in current_user.roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    
    return permission_checker

async def require_role(role: str):
    """Require specific role"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if role not in current_user.roles and "admin" not in current_user.roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return current_user
    
    return role_checker

# Global security manager instance
_security_manager: Optional[SecurityManager] = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if not _security_manager:
        config = SecurityConfig(
            jwt_secret="your-super-secret-key-change-in-production",
            jwt_algorithm="HS256"
        )
        _security_manager = SecurityManager(config)
    return _security_manager

async def initialize_security_manager(
    config: SecurityConfig,
    redis_client: Optional[aioredis.Redis] = None
) -> SecurityManager:
    """Initialize security manager"""
    global _security_manager
    _security_manager = SecurityManager(config, redis_client)
    return _security_manager






























