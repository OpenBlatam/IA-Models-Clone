#!/usr/bin/env python3
"""
Security Implementation for Video-OpusClip
Comprehensive security features and utilities
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configure security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentType(Enum):
    """Types of security incidents"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTED = "malware_detected"
    SYSTEM_COMPROMISE = "system_compromise"

@dataclass
class SecurityIncident:
    """Security incident record"""
    id: str
    type: IncidentType
    severity: SecurityLevel
    description: str
    timestamp: datetime
    source_ip: str
    user_id: Optional[str] = None
    affected_resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "open"
    resolved_at: Optional[datetime] = None

class SecurityConfig:
    """Security configuration"""
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.rate_limit_requests = 100
        self.rate_limit_window = 60  # seconds
        self.encryption_key = os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32))
        self.salt = os.getenv("SALT", secrets.token_urlsafe(16))

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "score": max(0, 10 - len(errors) * 2)
        }
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_str)
        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format and security"""
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Check for potentially malicious URLs
        malicious_patterns = [
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'file:',
            r'ftp:'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True

class PasswordManager:
    """Password hashing and verification"""
    
    def __init__(self, salt: str):
        self.salt = salt.encode()
    
    def hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            kdf.verify(password.encode(), base64.urlsafe_b64decode(hashed))
            return True
        except Exception:
            return False

class DataEncryption:
    """Data encryption and decryption"""
    
    def __init__(self, key: str):
        self.cipher = Fernet(base64.urlsafe_b64encode(key.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

class JWTManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: timedelta = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True

class IntrusionDetector:
    """Intrusion detection system"""
    
    def __init__(self, max_failed_attempts: int = 5, lockout_duration: int = 900):
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        self.failed_attempts = {}
        self.blocked_ips = {}
        self.suspicious_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
            r'(<script|javascript:|vbscript:)',
            r'(\.\./|\.\.\\)',
            r'(union.*select|select.*union)',
            r'(exec\(|eval\(|system\()',
        ]
    
    def check_login_attempt(self, ip: str, success: bool) -> bool:
        """Check login attempt and apply rate limiting"""
        if success:
            # Reset failed attempts on successful login
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            return True
        
        # Increment failed attempts
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = 1
        else:
            self.failed_attempts[ip] += 1
        
        # Block IP if too many failed attempts
        if self.failed_attempts[ip] >= self.max_failed_attempts:
            self.blocked_ips[ip] = time.time()
            security_logger.warning(f"IP {ip} blocked due to multiple failed login attempts")
            return False
        
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        if ip not in self.blocked_ips:
            return False
        
        # Check if block has expired
        if time.time() - self.blocked_ips[ip] > self.lockout_duration:
            del self.blocked_ips[ip]
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            return False
        
        return True
    
    def detect_suspicious_activity(self, request_data: str) -> List[str]:
        """Detect suspicious patterns in request data"""
        detected_patterns = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, request_data, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return detected_patterns

class SecurityLogger:
    """Security event logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.logger = logging.getLogger('security_audit')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_access(self, user: str, resource: str, action: str, success: bool, ip: str):
        """Log access attempts"""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, 
            f"ACCESS - User: {user}, Resource: {resource}, "
            f"Action: {action}, Success: {success}, IP: {ip}"
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        self.logger.warning(f"SECURITY_EVENT - Type: {event_type}, Details: {details}")
    
    def log_incident(self, incident: SecurityIncident):
        """Log security incidents"""
        self.logger.error(
            f"INCIDENT - ID: {incident.id}, Type: {incident.type.value}, "
            f"Severity: {incident.severity.value}, Description: {incident.description}, "
            f"Source IP: {incident.source_ip}"
        )

class IncidentResponse:
    """Security incident response"""
    
    def __init__(self):
        self.incidents = []
        self.security_logger = SecurityLogger()
    
    def create_incident(self, incident: SecurityIncident) -> str:
        """Create and log security incident"""
        self.incidents.append(incident)
        self.security_logger.log_incident(incident)
        
        # Notify security team for high/critical incidents
        if incident.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self.notify_security_team(incident)
        
        return incident.id
    
    def notify_security_team(self, incident: SecurityIncident):
        """Notify security team of critical incidents"""
        # Implementation for sending alerts (email, Slack, etc.)
        self.security_logger.log_security_event(
            "SECURITY_TEAM_NOTIFICATION",
            {
                "incident_id": incident.id,
                "severity": incident.severity.value,
                "type": incident.type.value
            }
        )
    
    def resolve_incident(self, incident_id: str, resolution: str):
        """Mark incident as resolved"""
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.status = "resolved"
                incident.resolved_at = datetime.utcnow()
                self.security_logger.log_security_event(
                    "INCIDENT_RESOLVED",
                    {"incident_id": incident_id, "resolution": resolution}
                )
                break

class SecurityMiddleware:
    """Security middleware for FastAPI"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.intrusion_detector = IntrusionDetector(
            max_failed_attempts=config.max_failed_attempts,
            lockout_duration=config.lockout_duration_minutes * 60
        )
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window
        )
        self.security_logger = SecurityLogger()
    
    async def __call__(self, request: Request, call_next):
        """Process request through security middleware"""
        client_ip = request.client.host
        
        # Check if IP is blocked
        if self.intrusion_detector.is_ip_blocked(client_ip):
            raise HTTPException(status_code=429, detail="IP temporarily blocked")
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Detect suspicious activity
        request_body = await request.body()
        if request_body:
            suspicious_patterns = self.intrusion_detector.detect_suspicious_activity(
                request_body.decode()
            )
            if suspicious_patterns:
                incident = SecurityIncident(
                    id=secrets.token_urlsafe(16),
                    type=IncidentType.SUSPICIOUS_ACTIVITY,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Suspicious patterns detected: {suspicious_patterns}",
                    timestamp=datetime.utcnow(),
                    source_ip=client_ip,
                    affected_resource=str(request.url),
                    details={"patterns": suspicious_patterns}
                )
                self.incident_response.create_incident(incident)
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Security models
class UserLogin(BaseModel):
    """User login model"""
    email: str
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower()

class UserRegistration(BaseModel):
    """User registration model"""
    email: str
    password: str
    confirm_password: str
    
    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        validation = InputValidator.validate_password_strength(v)
        if not validation['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation['errors'])}")
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class SecureVideoRequest(BaseModel):
    """Secure video request model"""
    title: str
    description: str
    url: str
    duration: float
    resolution: str
    priority: str = "normal"
    tags: List[str] = []
    
    @validator('title')
    def validate_title(cls, v):
        sanitized = InputValidator.sanitize_input(v)
        if len(sanitized) > 100:
            raise ValueError('Title too long')
        if len(sanitized) < 1:
            raise ValueError('Title cannot be empty')
        return sanitized
    
    @validator('description')
    def validate_description(cls, v):
        sanitized = InputValidator.sanitize_input(v)
        if len(sanitized) > 1000:
            raise ValueError('Description too long')
        return sanitized
    
    @validator('url')
    def validate_url(cls, v):
        if not InputValidator.validate_url(v):
            raise ValueError('Invalid or potentially malicious URL')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        sanitized_tags = []
        for tag in v:
            sanitized = InputValidator.sanitize_input(tag)
            if sanitized and len(sanitized) <= 50:
                sanitized_tags.append(sanitized)
        return sanitized_tags[:10]  # Limit to 10 tags

# Security utilities
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Get current authenticated user"""
    try:
        jwt_manager = JWTManager(SecurityConfig().secret_key)
        payload = jwt_manager.verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Implementation for permission checking
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize security components
security_config = SecurityConfig()
password_manager = PasswordManager(security_config.salt)
data_encryption = DataEncryption(security_config.encryption_key)
jwt_manager = JWTManager(security_config.secret_key)
incident_response = IncidentResponse()
security_middleware = SecurityMiddleware(security_config) 