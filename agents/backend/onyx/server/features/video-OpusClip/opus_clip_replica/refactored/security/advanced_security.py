"""
Advanced Security Features for Refactored Opus Clip

Comprehensive security implementation with:
- Authentication and authorization
- Rate limiting
- Input validation and sanitization
- Security headers
- Audit logging
- Encryption
- API security
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import asyncio
import hashlib
import hmac
import secrets
import time
import jwt
from datetime import datetime, timedelta
from functools import wraps
import structlog
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator
import redis
import bcrypt
from cryptography.fernet import Fernet
import re

logger = structlog.get_logger("advanced_security")

# Security configuration
class SecurityConfig:
    """Security configuration settings."""
    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        self.password_min_length = 8
        self.rate_limit_requests = 100
        self.rate_limit_window = 3600  # 1 hour
        self.max_file_size = 500 * 1024 * 1024  # 500MB
        self.allowed_file_types = ["video/mp4", "video/mov", "video/avi", "video/quicktime"]
        self.encryption_key = Fernet.generate_key()
        self.audit_log_enabled = True
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

# Initialize security config
security_config = SecurityConfig()

# Redis client for rate limiting and session storage
redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

# JWT token handler
class JWTHandler:
    """JWT token management."""
    
    @staticmethod
    def create_token(user_id: str, username: str, roles: List[str] = None) -> str:
        """Create JWT token."""
        payload = {
            "user_id": user_id,
            "username": username,
            "roles": roles or ["user"],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=security_config.jwt_expiry_hours)
        }
        return jwt.encode(payload, security_config.jwt_secret, algorithm=security_config.jwt_algorithm)
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, security_config.jwt_secret, algorithms=[security_config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    @staticmethod
    def refresh_token(token: str) -> str:
        """Refresh JWT token."""
        payload = JWTHandler.verify_token(token)
        return JWTHandler.create_token(
            payload["user_id"], 
            payload["username"], 
            payload.get("roles", ["user"])
        )

# Password hashing
class PasswordManager:
    """Password management utilities."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        errors = []
        
        if len(password) < security_config.password_min_length:
            errors.append(f"Password must be at least {security_config.password_min_length} characters long")
        
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength": PasswordManager._calculate_strength(password)
        }
    
    @staticmethod
    def _calculate_strength(password: str) -> str:
        """Calculate password strength."""
        score = 0
        
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if re.search(r"[A-Z]", password):
            score += 1
        if re.search(r"[a-z]", password):
            score += 1
        if re.search(r"\d", password):
            score += 1
        if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 4:
            return "medium"
        else:
            return "strong"

# Rate limiting
class RateLimiter:
    """Rate limiting implementation."""
    
    @staticmethod
    async def check_rate_limit(request: Request, identifier: str = None) -> bool:
        """Check if request is within rate limits."""
        if not identifier:
            identifier = request.client.host
        
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        window_start = current_time - security_config.rate_limit_window
        
        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, security_config.rate_limit_window)
        
        results = pipe.execute()
        current_requests = results[1]
        
        if current_requests >= security_config.rate_limit_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        return True
    
    @staticmethod
    async def get_rate_limit_info(identifier: str) -> Dict[str, Any]:
        """Get rate limit information for identifier."""
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        window_start = current_time - security_config.rate_limit_window
        
        # Get current request count
        current_requests = redis_client.zcount(key, window_start, current_time)
        
        return {
            "current_requests": current_requests,
            "limit": security_config.rate_limit_requests,
            "window_seconds": security_config.rate_limit_window,
            "reset_time": current_time + security_config.rate_limit_window
        }

# Input validation and sanitization
class InputValidator:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_video_file(file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video file upload."""
        errors = []
        
        # Check file size
        if file_info.get("size", 0) > security_config.max_file_size:
            errors.append(f"File size exceeds maximum allowed size of {security_config.max_file_size / (1024*1024):.1f}MB")
        
        # Check file type
        content_type = file_info.get("content_type", "")
        if content_type not in security_config.allowed_file_types:
            errors.append(f"File type {content_type} is not allowed")
        
        # Check filename
        filename = file_info.get("filename", "")
        if not re.match(r"^[a-zA-Z0-9._-]+$", filename):
            errors.append("Filename contains invalid characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def sanitize_string(input_string: str) -> str:
        """Sanitize string input."""
        if not input_string:
            return ""
        
        # Remove null bytes
        input_string = input_string.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        input_string = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', input_string)
        
        # Trim whitespace
        input_string = input_string.strip()
        
        return input_string
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None

# Encryption utilities
class EncryptionManager:
    """Encryption and decryption utilities."""
    
    def __init__(self):
        self.cipher = Fernet(security_config.encryption_key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_file(self, file_path: str, output_path: str) -> bool:
        """Encrypt file."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.cipher.encrypt(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return False
    
    def decrypt_file(self, encrypted_path: str, output_path: str) -> bool:
        """Decrypt file."""
        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False

# Audit logging
class AuditLogger:
    """Security audit logging."""
    
    @staticmethod
    async def log_security_event(event_type: str, user_id: str = None, 
                               details: Dict[str, Any] = None, request: Request = None):
        """Log security event."""
        if not security_config.audit_log_enabled:
            return
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": request.client.host if request else None,
            "user_agent": request.headers.get("user-agent") if request else None,
            "details": details or {}
        }
        
        # Log to structured logger
        logger.info("Security event", **log_entry)
        
        # Store in Redis for analysis
        try:
            redis_client.lpush("security_events", str(log_entry))
            redis_client.ltrim("security_events", 0, 10000)  # Keep last 10k events
        except Exception as e:
            logger.error(f"Failed to store security event: {e}")
    
    @staticmethod
    async def get_security_events(limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        try:
            events = redis_client.lrange("security_events", 0, limit - 1)
            return [eval(event) for event in events]
        except Exception as e:
            logger.error(f"Failed to retrieve security events: {e}")
            return []

# Security middleware
class SecurityMiddleware:
    """Security middleware for FastAPI."""
    
    @staticmethod
    def add_security_headers(request: Request, call_next):
        """Add security headers to response."""
        response = call_next(request)
        
        for header, value in security_config.security_headers.items():
            response.headers[header] = value
        
        return response
    
    @staticmethod
    def validate_request_size(request: Request, call_next):
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > security_config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail="Request entity too large"
            )
        
        return call_next(request)

# Authentication decorators
def require_auth(roles: List[str] = None):
    """Require authentication decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs
            request = kwargs.get('request')
            if not request:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Get token from Authorization header
            auth_header = request.headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")
            
            token = auth_header.split(" ")[1]
            
            try:
                # Verify token
                payload = JWTHandler.verify_token(token)
                
                # Check roles if specified
                if roles:
                    user_roles = payload.get("roles", [])
                    if not any(role in user_roles for role in roles):
                        raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Add user info to kwargs
                kwargs['current_user'] = payload
                
                return await func(*args, **kwargs)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                raise HTTPException(status_code=401, detail="Authentication failed")
        
        return wrapper
    return decorator

def require_rate_limit(identifier_func=None):
    """Require rate limiting decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                return await func(*args, **kwargs)
            
            # Get identifier for rate limiting
            if identifier_func:
                identifier = identifier_func(request)
            else:
                identifier = request.client.host
            
            # Check rate limit
            if not await RateLimiter.check_rate_limit(request, identifier):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Pydantic models for security
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        return InputValidator.sanitize_string(v)
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < security_config.password_min_length:
            raise ValueError(f'Password must be at least {security_config.password_min_length} characters long')
        return v

class RegisterRequest(BaseModel):
    """Registration request model."""
    username: str
    email: str
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return InputValidator.sanitize_string(v)
    
    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        validation_result = PasswordManager.validate_password_strength(v)
        if not validation_result['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['errors'])}")
        return v

class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        validation_result = PasswordManager.validate_password_strength(v)
        if not validation_result['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['errors'])}")
        return v

# Security utilities
class SecurityUtils:
    """General security utilities."""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def verify_csrf_token(token: str, session_token: str) -> bool:
        """Verify CSRF token."""
        return hmac.compare_digest(token, session_token)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        """Verify API key."""
        return hashlib.sha256(api_key.encode()).hexdigest() == hashed_key
    
    @staticmethod
    def is_safe_redirect_url(url: str) -> bool:
        """Check if redirect URL is safe."""
        if not url:
            return False
        
        # Only allow relative URLs or same origin
        if url.startswith('/') and not url.startswith('//'):
            return True
        
        # Check if URL is from same origin (simplified)
        return not url.startswith('http') or url.startswith('/')
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = filename.split('/')[-1]
        filename = filename.split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + ('.' + ext if ext else '')
        
        return filename

# Initialize encryption manager
encryption_manager = EncryptionManager()

# Security endpoints
class SecurityEndpoints:
    """Security-related API endpoints."""
    
    @staticmethod
    async def login(request: LoginRequest, http_request: Request):
        """User login endpoint."""
        # This would typically check against a user database
        # For demo purposes, we'll use a simple check
        
        # Log login attempt
        await AuditLogger.log_security_event(
            "login_attempt",
            request.username,
            {"ip": http_request.client.host},
            http_request
        )
        
        # In a real implementation, verify credentials against database
        # For now, we'll create a token for any valid username/password
        if len(request.password) >= security_config.password_min_length:
            token = JWTHandler.create_token(
                user_id="user_123",
                username=request.username,
                roles=["user"]
            )
            
            await AuditLogger.log_security_event(
                "login_success",
                request.username,
                {"ip": http_request.client.host},
                http_request
            )
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": security_config.jwt_expiry_hours * 3600
            }
        else:
            await AuditLogger.log_security_event(
                "login_failed",
                request.username,
                {"reason": "invalid_password", "ip": http_request.client.host},
                http_request
            )
            
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
    
    @staticmethod
    async def register(request: RegisterRequest, http_request: Request):
        """User registration endpoint."""
        # Log registration attempt
        await AuditLogger.log_security_event(
            "registration_attempt",
            request.username,
            {"email": request.email, "ip": http_request.client.host},
            http_request
        )
        
        # In a real implementation, create user in database
        # For now, we'll just return success
        
        await AuditLogger.log_security_event(
            "registration_success",
            request.username,
            {"email": request.email, "ip": http_request.client.host},
            http_request
        )
        
        return {
            "message": "User registered successfully",
            "username": request.username
        }
    
    @staticmethod
    async def get_security_events(limit: int = 100):
        """Get security events endpoint."""
        events = await AuditLogger.get_security_events(limit)
        return {
            "events": events,
            "total": len(events)
        }
    
    @staticmethod
    async def get_rate_limit_info(identifier: str):
        """Get rate limit information endpoint."""
        info = await RateLimiter.get_rate_limit_info(identifier)
        return info


