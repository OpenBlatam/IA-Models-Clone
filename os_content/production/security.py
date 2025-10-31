from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import hashlib
import hmac
import jwt
import bcrypt
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import get_production_config
            import os
        import html
        import re
        import re
        import uuid
        import os
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Production Security for OS Content UGC Video Generator
Authentication, authorization, and security measures
"""



logger = structlog.get_logger("os_content.security")

@dataclass
class User:
    """User data structure"""
    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    permissions: List[str] = None
    created_at: datetime = None
    last_login: datetime = None

@dataclass
class TokenData:
    """Token data structure"""
    user_id: str
    username: str
    permissions: List[str]
    exp: datetime

class SecurityManager:
    """Production security manager"""
    
    def __init__(self) -> Any:
        self.config = get_production_config()
        self.secret_key = self.config.secret_key
        self.jwt_secret = self.config.jwt_secret
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Rate limiting
        self.rate_limit_store = {}
        self.rate_limit_requests = self.config.rate_limit
        self.rate_limit_window = self.config.rate_limit_window
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # CORS configuration
        self.cors_origins = self.config.cors_origins
        self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.cors_headers = ["*"]
        
        # Password policy
        self.password_min_length = 8
        self.password_require_uppercase = True
        self.password_require_lowercase = True
        self.password_require_digits = True
        self.password_require_special = True
        
        # Session management
        self.max_sessions_per_user = 5
        self.session_timeout_minutes = 60
        
        # File upload security
        self.allowed_file_types = self.config.allowed_extensions
        self.max_file_size = self.config.max_file_size
        self.scan_uploads = True
        
        # API key management
        self.api_keys = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters long")
        
        if self.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.password_require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            username = payload.get("username")
            permissions = payload.get("permissions", [])
            exp = payload.get("exp")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            return TokenData(
                user_id=user_id,
                username=username,
                permissions=permissions,
                exp=datetime.fromtimestamp(exp)
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limit for client IP"""
        current_time = datetime.utcnow()
        
        if client_ip not in self.rate_limit_store:
            self.rate_limit_store[client_ip] = []
        
        # Remove old requests outside the window
        self.rate_limit_store[client_ip] = [
            req_time for req_time in self.rate_limit_store[client_ip]
            if current_time - req_time < timedelta(seconds=self.rate_limit_window)
        ]
        
        # Check if limit exceeded
        if len(self.rate_limit_store[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.rate_limit_store[client_ip].append(current_time)
        return True
    
    def get_rate_limit_info(self, client_ip: str) -> Dict[str, Any]:
        """Get rate limit information for client IP"""
        current_time = datetime.utcnow()
        
        if client_ip not in self.rate_limit_store:
            return {
                "remaining": self.rate_limit_requests,
                "reset_time": current_time + timedelta(seconds=self.rate_limit_window)
            }
        
        # Remove old requests
        self.rate_limit_store[client_ip] = [
            req_time for req_time in self.rate_limit_store[client_ip]
            if current_time - req_time < timedelta(seconds=self.rate_limit_window)
        ]
        
        remaining = max(0, self.rate_limit_requests - len(self.rate_limit_store[client_ip]))
        
        return {
            "remaining": remaining,
            "reset_time": current_time + timedelta(seconds=self.rate_limit_window)
        }
    
    async def validate_file_upload(self, filename: str, file_size: int, content_type: str) -> Dict[str, Any]:
        """Validate file upload"""
        errors = []
        
        # Check file size
        if file_size > self.max_file_size:
            errors.append(f"File size exceeds maximum allowed size of {self.max_file_size} bytes")
        
        # Check file extension
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        if file_extension not in [ext.replace('.', '') for ext in self.allowed_file_types]:
            errors.append(f"File type '{file_extension}' is not allowed")
        
        # Check content type
        allowed_content_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav'
        }
        
        if file_extension in allowed_content_types:
            expected_content_type = allowed_content_types[file_extension]
            if content_type != expected_content_type:
                errors.append(f"Content type '{content_type}' does not match file extension")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def scan_file_for_malware(self, file_path: str) -> bool:
        """Scan file for malware (placeholder implementation)"""
        if not self.scan_uploads:
            return True
        
        try:
            # This would integrate with actual antivirus software
            # For now, just check file size and basic validation
            file_size = os.path.getsize(file_path)
            
            # Basic checks
            if file_size == 0:
                return False
            
            # Check for suspicious patterns (very basic)
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read(1024)  # Read first 1KB
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if b'\x00\x00\x00\x00' * 100 in content:  # Suspicious null bytes
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"File scan failed: {e}")
            return False
    
    async def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        
        logger.info(f"API key generated for user {user_id}")
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        if api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        key_data["last_used"] = datetime.utcnow()
        
        return key_data
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"API key revoked: {api_key[:8]}...")
            return True
        return False
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or "admin" in user_permissions
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input"""
        
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove other potentially dangerous tags
        dangerous_tags = ['iframe', 'object', 'embed', 'form', 'input', 'textarea', 'select']
        for tag in dangerous_tags:
            sanitized = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def generate_secure_filename(self, original_filename: str) -> str:
        """Generate secure filename"""
        
        # Get file extension
        _, ext = os.path.splitext(original_filename)
        
        # Generate secure filename
        secure_filename = f"{uuid.uuid4().hex}{ext}"
        
        return secure_filename
    
    def create_password_reset_token(self, user_id: str) -> str:
        """Create password reset token"""
        data = {
            "user_id": user_id,
            "type": "password_reset",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(data, self.jwt_secret, algorithm=self.algorithm)
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            
            if payload.get("type") != "password_reset":
                return None
            
            return payload.get("user_id")
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent")
        }
        
        logger.warning(f"Security event: {event_type}", **event)
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers"""
        return self.security_headers.copy()
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "origins": self.cors_origins,
            "methods": self.cors_methods,
            "headers": self.cors_headers,
            "allow_credentials": True
        }
    
    def cleanup_expired_tokens(self) -> Any:
        """Cleanup expired tokens and sessions"""
        current_time = datetime.utcnow()
        
        # Cleanup rate limit store
        for client_ip in list(self.rate_limit_store.keys()):
            self.rate_limit_store[client_ip] = [
                req_time for req_time in self.rate_limit_store[client_ip]
                if current_time - req_time < timedelta(seconds=self.rate_limit_window)
            ]
            
            if not self.rate_limit_store[client_ip]:
                del self.rate_limit_store[client_ip]
        
        # Cleanup old API keys (older than 30 days)
        for api_key in list(self.api_keys.keys()):
            key_data = self.api_keys[api_key]
            if current_time - key_data["created_at"] > timedelta(days=30):
                del self.api_keys[api_key]
                logger.info(f"Expired API key removed: {api_key[:8]}...")

# Security dependencies
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current user from JWT token"""
    security_manager = SecurityManager()
    return security_manager.verify_token(credentials.credentials)

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_dependency(current_user: TokenData = Depends(get_current_user)):
        security_manager = SecurityManager()
        if not security_manager.check_permission(current_user.permissions, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_dependency

def require_admin(current_user: TokenData = Depends(get_current_user)):
    """Require admin permissions"""
    security_manager = SecurityManager()
    if not security_manager.check_permission(current_user.permissions, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Global security manager instance
security_manager = SecurityManager() 