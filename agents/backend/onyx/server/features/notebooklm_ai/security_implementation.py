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

import asyncio
import time
import logging
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime, timezone, timedelta
from functools import wraps
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, field_validator
import jwt
import redis.asyncio as redis
from databases import Database
from sqlalchemy import text
import aiohttp
from prometheus_client import Counter, Histogram, Gauge
import structlog
from typing import Any, List, Dict, Optional
"""
Security Implementation - Practical Application of Key Principles
Implements security best practices for the notebooklm_ai FastAPI application
"""



# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

class SecurityConfig:
    """Security configuration with best practices."""
    
    def __init__(self) -> Any:
        self.jwt_secret = "your-super-secret-jwt-key-change-in-production"
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = 3600  # 1 hour
        self.password_min_length = 12
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.rate_limit_per_minute = 60
        self.session_timeout = 3600
        self.allowed_file_types = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.encryption_key = "your-32-byte-encryption-key-here"

# ============================================================================
# SECURITY MODELS
# ============================================================================

class UserLogin(BaseModel):
    """Secure login model with validation."""
    model_config = ConfigDict(extra="forbid")
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    mfa_code: Optional[str] = Field(None, max_length=10, description="MFA code")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        # Only allow alphanumeric and underscore
        if not v.replace('_', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()

class SecurityEvent(BaseModel):
    """Security event logging model."""
    model_config = ConfigDict(extra="forbid")
    
    event_type: str = Field(..., description="Type of security event")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    ip_address: str = Field(..., description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    action: str = Field(..., description="Action performed")
    result: str = Field(..., description="Result of action")
    risk_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Risk score")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")

# ============================================================================
# SECURITY SERVICES
# ============================================================================

class SecurityService:
    """Core security service implementing key principles."""
    
    def __init__(self, config: SecurityConfig, redis_client: redis.Redis, db: Database):
        
    """__init__ function."""
self.config = config
        self.redis = redis_client
        self.db = db
        self.logger = structlog.get_logger()
        
        # Security metrics
        self.login_attempts = Counter('security_login_attempts_total', 'Total login attempts', ['result'])
        self.failed_logins = Counter('security_failed_logins_total', 'Failed login attempts', ['username'])
        self.security_events = Counter('security_events_total', 'Security events', ['type', 'severity'])
        self.active_sessions = Gauge('security_active_sessions', 'Number of active sessions')
    
    async def authenticate_user(self, username: str, password: str, mfa_code: Optional[str] = None) -> Dict:
        """Authenticate user with multiple security checks."""
        start_time = time.time()
        
        try:
            # Check rate limiting
            await self.check_rate_limit(username)
            
            # Check account lockout
            if await self.is_account_locked(username):
                await self.log_security_event(SecurityEvent(
                    event_type="account_locked",
                    user_id=username,
                    ip_address="unknown",
                    action="login_attempt",
                    result="blocked",
                    risk_score=8.0,
                    details={"reason": "account_locked"}
                ))
                raise HTTPException(status_code=423, detail="Account temporarily locked")
            
            # Verify credentials
            user = await self.verify_credentials(username, password)
            if not user:
                await self.record_failed_login(username)
                await self.log_security_event(SecurityEvent(
                    event_type="failed_login",
                    user_id=username,
                    ip_address="unknown",
                    action="login_attempt",
                    result="failed",
                    risk_score=5.0,
                    details={"reason": "invalid_credentials"}
                ))
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Verify MFA if required
            if user.get('mfa_enabled') and mfa_code:
                if not await self.verify_mfa(username, mfa_code):
                    await self.log_security_event(SecurityEvent(
                        event_type="mfa_failure",
                        user_id=username,
                        ip_address="unknown",
                        action="mfa_verification",
                        result="failed",
                        risk_score=7.0
                    ))
                    raise HTTPException(status_code=401, detail="Invalid MFA code")
            
            # Create session
            session_id = await self.create_secure_session(user['id'])
            
            # Log successful login
            await self.log_security_event(SecurityEvent(
                event_type="successful_login",
                user_id=username,
                ip_address="unknown",
                action="login_attempt",
                result="success",
                risk_score=1.0,
                details={"session_id": session_id}
            ))
            
            self.login_attempts.labels(result="success").inc()
            
            return {
                "user_id": user['id'],
                "username": user['username'],
                "session_id": session_id,
                "access_token": await self.create_jwt_token(user['id']),
                "expires_in": self.config.jwt_expiration
            }
            
        except HTTPException:
            self.login_attempts.labels(result="failure").inc()
            raise
        except Exception as e:
            self.logger.error("Authentication error", error=str(e), username=username)
            raise HTTPException(status_code=500, detail="Authentication failed")
    
    async def verify_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Verify user credentials securely."""
        query = """
            SELECT id, username, password_hash, mfa_enabled, is_active
            FROM users WHERE username = :username
        """
        
        result = await self.db.fetch_one(query, {"username": username})
        if not result:
            return None
        
        user = dict(result)
        
        # Check if account is active
        if not user['is_active']:
            return None
        
        # Verify password (in production, use proper password hashing)
        if user['password_hash'] == hashlib.sha256(password.encode()).hexdigest():
            return user
        
        return None
    
    async def check_rate_limit(self, username: str):
        """Check rate limiting for login attempts."""
        key = f"rate_limit:login:{username}"
        current_count = await self.redis.get(key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= self.config.max_login_attempts:
            raise HTTPException(status_code=429, detail="Too many login attempts")
        
        await self.redis.setex(key, self.config.lockout_duration, str(current_count + 1))
    
    async def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        key = f"account_locked:{username}"
        return await self.redis.exists(key) > 0
    
    async def record_failed_login(self, username: str):
        """Record failed login attempt."""
        key = f"failed_logins:{username}"
        failed_count = await self.redis.get(key)
        failed_count = int(failed_count) if failed_count else 0
        
        failed_count += 1
        await self.redis.setex(key, self.config.lockout_duration, str(failed_count))
        
        self.failed_logins.labels(username=username).inc()
        
        # Lock account if too many failed attempts
        if failed_count >= self.config.max_login_attempts:
            await self.redis.setex(f"account_locked:{username}", self.config.lockout_duration, "1")
    
    async def verify_mfa(self, username: str, mfa_code: str) -> bool:
        """Verify MFA code (simplified implementation)."""
        # In production, use proper TOTP implementation
        stored_code = await self.redis.get(f"mfa:{username}")
        return stored_code and stored_code.decode() == mfa_code
    
    async def create_secure_session(self, user_id: str) -> str:
        """Create secure session with proper management."""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "ip_address": "unknown",  # Would be set from request
            "user_agent": "unknown"   # Would be set from request
        }
        
        await self.redis.setex(
            f"session:{session_id}",
            self.config.session_timeout,
            json.dumps(session_data)
        )
        
        self.active_sessions.inc()
        return session_id
    
    async def create_jwt_token(self, user_id: str) -> str:
        """Create JWT token with security best practices."""
        payload = {
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(seconds=self.config.jwt_expiration),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),  # Unique token ID
            "type": "access"
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token securely."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            # Check if token is expired
            if datetime.fromtimestamp(payload['exp'], tz=timezone.utc) < datetime.now(timezone.utc):
                raise jwt.ExpiredSignatureError
            
            return payload
        except jwt.ExpiredSignatureError:
            await self.log_security_event(SecurityEvent(
                event_type="token_expired",
                action="token_verification",
                result="failed",
                risk_score=3.0,
                details={"reason": "expired"}
            ))
            return None
        except jwt.JWTError:
            await self.log_security_event(SecurityEvent(
                event_type="invalid_token",
                action="token_verification",
                result="failed",
                risk_score=5.0,
                details={"reason": "invalid"}
            ))
            return None
    
    async def log_security_event(self, event: SecurityEvent):
        """Log security event with structured logging."""
        self.logger.info(
            "security_event",
            event_type=event.event_type,
            user_id=event.user_id,
            ip_address=event.ip_address,
            action=event.action,
            result=event.result,
            risk_score=event.risk_score,
            details=event.details
        )
        
        # Store in database
        query = """
            INSERT INTO security_events (
                event_type, user_id, ip_address, user_agent, action, 
                result, risk_score, details, timestamp
            ) VALUES (
                :event_type, :user_id, :ip_address, :user_agent, :action,
                :result, :risk_score, :details, :timestamp
            )
        """
        
        await self.db.execute(query, {
            "event_type": event.event_type,
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "action": event.action,
            "result": event.result,
            "risk_score": event.risk_score,
            "details": json.dumps(event.details),
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Update metrics
        severity = "high" if event.risk_score > 7 else "medium" if event.risk_score > 4 else "low"
        self.security_events.labels(type=event.event_type, severity=severity).inc()

# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

class SecurityMiddleware:
    """Security middleware implementing defense in depth."""
    
    def __init__(self, app, security_service: SecurityService):
        
    """__init__ function."""
self.app = app
        self.security_service = security_service
        self.logger = structlog.get_logger()
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = time.time()
            request_id = scope.get("request_id", "unknown")
            
            # Extract request information
            method = scope["method"]
            path = scope["path"]
            client_ip = scope.get("client", ["unknown"])[0]
            
            # Security checks
            try:
                # Check for suspicious patterns
                if await self.detect_suspicious_activity(method, path, client_ip):
                    await self.security_service.log_security_event(SecurityEvent(
                        event_type="suspicious_activity",
                        ip_address=client_ip,
                        action=f"{method} {path}",
                        result="blocked",
                        risk_score=8.0,
                        details={"reason": "suspicious_pattern"}
                    ))
                    
                    # Return 403 Forbidden
                    await send({
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"application/json")]
                    })
                    await send({
                        "type": "http.response.body",
                        "body": json.dumps({"detail": "Access denied"}).encode()
                    })
                    return
                
                # Add security headers
                async def send_with_security_headers(message) -> Any:
                    if message["type"] == "http.response.start":
                        security_headers = [
                            (b"x-content-type-options", b"nosniff"),
                            (b"x-frame-options", b"DENY"),
                            (b"x-xss-protection", b"1; mode=block"),
                            (b"referrer-policy", b"strict-origin-when-cross-origin"),
                            (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                            (b"content-security-policy", b"default-src 'self'; script-src 'self' 'unsafe-inline'"),
                            (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                        ]
                        
                        for header_name, header_value in security_headers:
                            message["headers"].append((header_name, header_value))
                    
                    await send(message)
                
                await self.app(scope, receive, send_with_security_headers)
                
            except Exception as e:
                self.logger.error("Security middleware error", error=str(e), request_id=request_id)
                await self.app(scope, receive, send)
    
    async def detect_suspicious_activity(self, method: str, path: str, client_ip: str) -> bool:
        """Detect suspicious activity patterns."""
        # Check for SQL injection attempts
        sql_patterns = ["'", "1=1", "DROP", "DELETE", "INSERT", "UPDATE", "UNION"]
        if any(pattern.lower() in path.lower() for pattern in sql_patterns):
            return True
        
        # Check for XSS attempts
        xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
        if any(pattern.lower() in path.lower() for pattern in xss_patterns):
            return True
        
        # Check for path traversal
        if ".." in path or "~" in path:
            return True
        
        # Check for excessive requests
        request_count = await self.security_service.redis.get(f"requests:{client_ip}")
        if request_count and int(request_count) > 1000:  # More than 1000 requests
            return True
        
        return False

# ============================================================================
# SECURITY DEPENDENCIES
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    security_service: SecurityService = Depends()
) -> Dict:
    """Get current user with security validation."""
    payload = await security_service.verify_jwt_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

async def require_admin(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Require admin privileges."""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return current_user

# ============================================================================
# SECURITY ROUTES
# ============================================================================

class SecurityRoutes:
    """Security-related API routes."""
    
    def __init__(self, security_service: SecurityService):
        
    """__init__ function."""
self.security_service = security_service
        self.logger = structlog.get_logger()
    
    async def login(self, login_data: UserLogin, request: Request) -> Dict:
        """Secure login endpoint."""
        # Add IP address and user agent to security event
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        result = await self.security_service.authenticate_user(
            login_data.username, 
            login_data.password, 
            login_data.mfa_code
        )
        
        # Update security event with client information
        await self.security_service.log_security_event(SecurityEvent(
            event_type="successful_login",
            user_id=login_data.username,
            ip_address=client_ip,
            user_agent=user_agent,
            action="login",
            result="success",
            risk_score=1.0,
            details={"session_id": result["session_id"]}
        ))
        
        return result
    
    async def logout(self, current_user: Dict = Depends(get_current_user)) -> Dict:
        """Secure logout endpoint."""
        # Invalidate session
        session_id = current_user.get('session_id')
        if session_id:
            await self.security_service.redis.delete(f"session:{session_id}")
            self.security_service.active_sessions.dec()
        
        await self.security_service.log_security_event(SecurityEvent(
            event_type="logout",
            user_id=current_user.get('user_id'),
            ip_address="unknown",
            action="logout",
            result="success",
            risk_score=1.0
        ))
        
        return {"message": "Logged out successfully"}
    
    async def get_security_events(
        self, 
        admin_user: Dict = Depends(require_admin),
        limit: int = 100
    ) -> List[Dict]:
        """Get security events (admin only)."""
        query = """
            SELECT * FROM security_events 
            ORDER BY timestamp DESC 
            LIMIT :limit
        """
        
        events = await self.security_service.db.fetch_all(query, {"limit": limit})
        return [dict(event) for event in events]

# ============================================================================
# SECURITY UTILITIES
# ============================================================================

class SecurityUtils:
    """Utility functions for security operations."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password securely."""
        # In production, use bcrypt or Argon2
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def generate_secure_token() -> str:
        """Generate secure random token."""
        return hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    
    @staticmethod
    async def validate_file_upload(filename: str, content_type: str, file_size: int) -> bool:
        """Validate file upload security."""
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        file_ext = '.' + filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            return False
        
        # Check content type
        allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
        if content_type not in allowed_types:
            return False
        
        # Check file size (10MB limit)
        if file_size > 10 * 1024 * 1024:
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input."""
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')']
        sanitized = input_str
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()

# ============================================================================
# SECURITY DECORATORS
# ============================================================================

def require_authentication(func) -> Any:
    """Decorator to require authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Authentication logic would be implemented here
        return await func(*args, **kwargs)
    return wrapper

def rate_limit(requests_per_minute: int = 60):
    """Decorator to implement rate limiting."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Rate limiting logic would be implemented here
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def audit_log(action: str):
    """Decorator to log security events."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Audit logging logic would be implemented here
            return await func(*args, **kwargs)
        return wrapper
    return decorator 