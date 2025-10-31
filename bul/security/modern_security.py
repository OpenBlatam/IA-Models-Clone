"""
Modern Security System for BUL
==============================

Enhanced security with modern libraries and best practices.
"""

import hashlib
import secrets
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from passlib.context import CryptContext
from passlib.hash import argon2, bcrypt
from jose import JWTError, jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from ..config.modern_config import get_config
from ..utils.modern_logging import log_security_event

class ModernPasswordManager:
    """Modern password management with multiple hashing algorithms"""
    
    def __init__(self):
        self.config = get_config()
        # Use Argon2 as primary, bcrypt as fallback
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__rounds=3,
            argon2__memory_cost=65536,
            argon2__parallelism=4,
            bcrypt__rounds=12
        )
    
    def hash_password(self, password: str) -> str:
        """Hash a password using the best available algorithm"""
        if len(password) < self.config.security.password_min_length:
            raise ValueError(f"Password must be at least {self.config.security.password_min_length} characters")
        
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            log_security_event("password_verification_error", str(e))
            return False
    
    def needs_update(self, hashed_password: str) -> bool:
        """Check if password hash needs updating"""
        return self.pwd_context.needs_update(hashed_password)

class ModernJWTManager:
    """Modern JWT token management"""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.secret_key
        self.algorithm = self.config.security.algorithm
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create an access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.config.security.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            log_security_event("jwt_creation_error", str(e))
            raise
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.security.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            log_security_event("jwt_refresh_creation_error", str(e))
            raise
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                log_security_event("invalid_token_type", f"Expected {token_type}, got {payload.get('type')}")
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                log_security_event("token_expired", f"Token expired at {datetime.fromtimestamp(exp)}")
                return None
            
            return payload
        except JWTError as e:
            log_security_event("jwt_verification_error", str(e))
            return None
        except Exception as e:
            log_security_event("jwt_unexpected_error", str(e))
            return None

class ModernEncryption:
    """Modern encryption utilities"""
    
    def __init__(self):
        self.config = get_config()
        self._fernet = None
    
    def _get_fernet(self) -> Fernet:
        """Get or create Fernet instance"""
        if self._fernet is None:
            # Derive key from secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'bul_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.security.secret_key.encode()))
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            fernet = self._get_fernet()
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            log_security_event("encryption_error", str(e))
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            fernet = self._get_fernet()
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            log_security_event("decryption_error", str(e))
            raise

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.config = get_config()
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed based on rate limit"""
        now = time.time()
        window_start = now - self.config.security.rate_limit_window
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.config.security.rate_limit_requests:
            log_security_event("rate_limit_exceeded", f"Identifier: {identifier}")
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        window_start = now - self.config.security.rate_limit_window
        
        if identifier in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            return max(0, self.config.security.rate_limit_requests - len(recent_requests))
        
        return self.config.security.rate_limit_requests

class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 20:
            return False
        
        # Check for common patterns
        if api_key.startswith('sk-') or api_key.startswith('Bearer '):
            return True
        
        # Check for alphanumeric with some special chars
        if len(api_key) >= 32 and any(c.isalnum() for c in api_key):
            return True
        
        return False
    
    @staticmethod
    def validate_input(input_str: str, max_length: int = 1000) -> bool:
        """Validate user input"""
        if not input_str or len(input_str) > max_length:
            return False
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            '<script', 'javascript:', 'data:', 'vbscript:',
            'onload=', 'onerror=', 'onclick=', 'onmouseover='
        ]
        
        input_lower = input_str.lower()
        for pattern in dangerous_patterns:
            if pattern in input_lower:
                log_security_event("dangerous_input_detected", f"Pattern: {pattern}")
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized

class SecurityHeaders:
    """Security headers management"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

# Global security instances
_password_manager: Optional[ModernPasswordManager] = None
_jwt_manager: Optional[ModernJWTManager] = None
_encryption: Optional[ModernEncryption] = None
_rate_limiter: Optional[RateLimiter] = None

def get_password_manager() -> ModernPasswordManager:
    """Get the global password manager"""
    global _password_manager
    if _password_manager is None:
        _password_manager = ModernPasswordManager()
    return _password_manager

def get_jwt_manager() -> ModernJWTManager:
    """Get the global JWT manager"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = ModernJWTManager()
    return _jwt_manager

def get_encryption() -> ModernEncryption:
    """Get the global encryption instance"""
    global _encryption
    if _encryption is None:
        _encryption = ModernEncryption()
    return _encryption

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter




