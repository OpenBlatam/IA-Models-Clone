#!/usr/bin/env python3
"""
Modular Security Implementation for Video-OpusClip
Iterative approach with reusable components to avoid code duplication
"""

import asyncio
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from functools import wraps, partial
from dataclasses import dataclass
from enum import Enum

import jwt
from cryptography.fernet import Fernet
from fastapi import HTTPException

# Security configuration
@dataclass
class SecurityConfig:
    """Centralized security configuration"""
    secret_key: str = "your-secret-key-change-this"
    encryption_key: str = "your-encryption-key-change-this"
    salt: str = "your-salt-change-this"
    max_login_attempts: int = 5
    lockout_duration: int = 900
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    jwt_expire_minutes: int = 30

# Security enums
class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationType(Enum):
    EMAIL = "email"
    PASSWORD = "password"
    URL = "url"
    INPUT = "input"

# Modular validation functions
class Validator:
    """Modular validation system"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        import re
        validations = [
            (len(password) >= 8, "Password must be at least 8 characters"),
            (re.search(r'[A-Z]', password), "Password must contain uppercase letter"),
            (re.search(r'[a-z]', password), "Password must contain lowercase letter"),
            (re.search(r'\d', password), "Password must contain digit"),
            (re.search(r'[!@#$%^&*(),.?":{}|<>]', password), "Password must contain special character")
        ]
        
        errors = [msg for check, msg in validations if not check]
        return {
            "valid": not errors,
            "errors": errors,
            "score": max(0, 10 - len(errors) * 2)
        }
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL security"""
        import re
        if not url.startswith(('http://', 'https://')):
            return False
        
        malicious_patterns = [
            r'javascript:', r'data:', r'vbscript:', r'file:', r'ftp:'
        ]
        
        return not any(re.search(pattern, url, re.IGNORECASE) for pattern in malicious_patterns)
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        import re
        sanitized = re.sub(r'[<>"\']', '', text)
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()
    
    @classmethod
    def validate_all(cls, data: Dict[str, Any], validations: Dict[str, ValidationType]) -> Dict[str, Any]:
        """Validate multiple fields at once"""
        results = {}
        for field, validation_type in validations.items():
            value = data.get(field, "")
            
            if validation_type == ValidationType.EMAIL:
                results[field] = cls.validate_email(value)
            elif validation_type == ValidationType.PASSWORD:
                results[field] = cls.validate_password(value)
            elif validation_type == ValidationType.URL:
                results[field] = cls.validate_url(value)
            elif validation_type == ValidationType.INPUT:
                results[field] = cls.sanitize_input(value)
        
        return results

# Modular encryption system
class CryptoManager:
    """Modular encryption/decryption system"""
    
    def __init__(self, key: str):
        self.cipher = Fernet(key.encode())
    
    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: List[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], fields_to_decrypt: List[str]) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
        return decrypted_data

# Modular JWT system
class JWTManager:
    """Modular JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, data: Dict[str, Any], expires_minutes: int = 30) -> str:
        """Create JWT token"""
        payload = data.copy()
        payload.update({
            "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
            "iat": datetime.utcnow()
        })
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def create_tokens(self, data: Dict[str, Any], access_expires: int = 30, refresh_expires: int = 1440) -> Dict[str, str]:
        """Create both access and refresh tokens"""
        return {
            "access_token": self.create_token(data, access_expires),
            "refresh_token": self.create_token({**data, "type": "refresh"}, refresh_expires)
        }

# Modular rate limiting system
class RateLimiter:
    """Modular rate limiting system"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        if identifier not in self.requests:
            return self.max_requests
        
        valid_requests = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(valid_requests))

# Modular intrusion detection
class IntrusionDetector:
    """Modular intrusion detection system"""
    
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
        """Check login attempt"""
        if success:
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            return True
        
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = 1
        else:
            self.failed_attempts[ip] += 1
        
        if self.failed_attempts[ip] >= self.max_failed_attempts:
            self.blocked_ips[ip] = time.time()
            return False
        
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip not in self.blocked_ips:
            return False
        
        if time.time() - self.blocked_ips[ip] > self.lockout_duration:
            del self.blocked_ips[ip]
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            return False
        
        return True
    
    def detect_suspicious_activity(self, data: str) -> List[str]:
        """Detect suspicious patterns"""
        import re
        detected = []
        for pattern in self.suspicious_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                detected.append(pattern)
        return detected

# Modular logging system
class SecurityLogger:
    """Modular security logging system"""
    
    def __init__(self, log_file: str = "security.log"):
        self.log_file = log_file
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        print(f"SECURITY_LOG: {log_entry}")
    
    def log_access(self, user: str, resource: str, action: str, success: bool, ip: str) -> None:
        """Log access attempt"""
        self.log_event("ACCESS", {
            "user": user,
            "resource": resource,
            "action": action,
            "success": success,
            "ip": ip
        })
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event"""
        self.log_event("SECURITY_EVENT", details)

# Modular decorator system
class SecurityDecorators:
    """Modular security decorators"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.jwt_manager = JWTManager(config.secret_key)
        self.rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        self.intrusion_detector = IntrusionDetector(config.max_login_attempts, config.lockout_duration)
        self.logger = SecurityLogger()
    
    def require_auth(self, func: Callable) -> Callable:
        """Authentication decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token = kwargs.get('token')
            if not token:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            try:
                user = self.jwt_manager.verify_token(token)
                kwargs['current_user'] = user
                return await func(*args, **kwargs)
            except HTTPException:
                raise HTTPException(status_code=401, detail="Invalid authentication")
        
        return wrapper
    
    def rate_limit(self, max_requests: int = None, window: int = None) -> Callable:
        """Rate limiting decorator"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                limiter = RateLimiter(
                    max_requests or self.config.rate_limit_requests,
                    window or self.config.rate_limit_window
                )
                
                identifier = kwargs.get('client_ip', 'unknown')
                if not limiter.is_allowed(identifier):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def validate_input(self, validation_rules: Dict[str, ValidationType]) -> Callable:
        """Input validation decorator"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if 'data' in kwargs:
                    validation_results = Validator.validate_all(kwargs['data'], validation_rules)
                    kwargs['validation_results'] = validation_results
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Modular user management
class UserManager:
    """Modular user management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.crypto = CryptoManager(config.encryption_key)
        self.jwt_manager = JWTManager(config.secret_key)
        self.validator = Validator()
        self.logger = SecurityLogger()
        self.users = {}  # In production, use database
    
    def register_user(self, email: str, password: str, client_ip: str) -> Dict[str, Any]:
        """Register user with validation"""
        # Validate input
        if not self.validator.validate_email(email):
            raise ValueError("Invalid email format")
        
        if email in self.users:
            raise ValueError("User already exists")
        
        password_validation = self.validator.validate_password(password)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {password_validation['errors']}")
        
        # Create user
        import hashlib
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), self.config.salt.encode(), 100000).hex()
        
        self.users[email] = {
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "permissions": ["user"]
        }
        
        self.logger.log_access(email, "/auth/register", "register", True, client_ip)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "data": {"email": email}
        }
    
    def authenticate_user(self, email: str, password: str, client_ip: str) -> Optional[Dict[str, str]]:
        """Authenticate user"""
        # Check if user exists
        if email not in self.users:
            self.logger.log_access("unknown", "/auth/login", "login", False, client_ip)
            return None
        
        # Verify password
        import hashlib
        user = self.users[email]
        hashed_input = hashlib.pbkdf2_hmac('sha256', password.encode(), self.config.salt.encode(), 100000).hex()
        
        if hashed_input != user["hashed_password"]:
            self.logger.log_access(email, "/auth/login", "login", False, client_ip)
            return None
        
        # Success
        self.logger.log_access(email, "/auth/login", "login", True, client_ip)
        
        # Generate tokens
        tokens = self.jwt_manager.create_tokens({
            "sub": email,
            "permissions": user["permissions"]
        })
        
        return tokens

# Modular video processing
class VideoProcessor:
    """Modular video processing with security"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.crypto = CryptoManager(config.encryption_key)
        self.validator = Validator()
        self.logger = SecurityLogger()
    
    def process_video(self, video_data: Dict[str, Any], user: Dict[str, Any], client_ip: str) -> Dict[str, Any]:
        """Process video with security validation"""
        # Validate input
        validation_rules = {
            "title": ValidationType.INPUT,
            "url": ValidationType.URL,
            "description": ValidationType.INPUT
        }
        
        validation_results = Validator.validate_all(video_data, validation_rules)
        
        # Check validation results
        for field, result in validation_results.items():
            if isinstance(result, dict) and not result.get("valid", True):
                raise ValueError(f"Validation failed for {field}: {result.get('errors', [])}")
        
        # Encrypt sensitive data
        fields_to_encrypt = ["description"]
        encrypted_data = self.crypto.encrypt_dict(video_data, fields_to_encrypt)
        
        # Add metadata
        result = {
            **encrypted_data,
            "id": secrets.token_urlsafe(16),
            "user": user.get("sub"),
            "processed_at": datetime.utcnow(),
            "validation_results": validation_results
        }
        
        self.logger.log_access(user.get("sub"), "/videos", "process", True, client_ip)
        
        return result

# Example usage with modular components
async def main():
    """Example usage of modular security system"""
    print("🔒 Modular Security System Example")
    
    # Initialize configuration
    config = SecurityConfig()
    
    # Initialize components
    user_manager = UserManager(config)
    video_processor = VideoProcessor(config)
    decorators = SecurityDecorators(config)
    
    # Register user
    try:
        result = user_manager.register_user("user@example.com", "SecurePass123!", "127.0.0.1")
        print("✅ User registered")
    except ValueError as e:
        print(f"❌ Registration failed: {e}")
    
    # Authenticate user
    tokens = user_manager.authenticate_user("user@example.com", "SecurePass123!", "127.0.0.1")
    if tokens:
        print(f"✅ User authenticated, access token: {tokens['access_token'][:20]}...")
    
    # Process video
    video_data = {
        "title": "My Video",
        "url": "https://example.com/video.mp4",
        "description": "A great video with sensitive information"
    }
    
    user_data = {"sub": "user@example.com", "permissions": ["user"]}
    processed_video = video_processor.process_video(video_data, user_data, "127.0.0.1")
    
    print(f"✅ Video processed: {processed_video['id']}")
    print(f"   Title: {processed_video['title']}")
    print(f"   Encrypted description: {processed_video['description'][:20]}...")
    
    # Demonstrate modular decorators
    @decorators.require_auth
    @decorators.rate_limit(max_requests=10, window=60)
    @decorators.validate_input({
        "title": ValidationType.INPUT,
        "url": ValidationType.URL
    })
    async def secure_endpoint(data: Dict, current_user: Dict, validation_results: Dict):
        return {"success": True, "data": data, "validation": validation_results}
    
    print("🎯 Modular security system ready!")

if __name__ == "__main__":
    asyncio.run(main()) 