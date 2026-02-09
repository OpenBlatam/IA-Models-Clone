#!/usr/bin/env python3
"""
Functional Security Implementation for Video-OpusClip
Declarative programming approach without classes
"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from functools import wraps, partial

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import HTTPException, Depends, Request
from pydantic import BaseModel, validator

# Configuration
SECURITY_CONFIG = {
    "secret_key": "your-secret-key-change-this",
    "encryption_key": "your-encryption-key-change-this", 
    "salt": "your-salt-change-this",
    "max_login_attempts": 5,
    "lockout_duration": 900,
    "rate_limit_requests": 100,
    "rate_limit_window": 60,
    "jwt_expire_minutes": 30
}

# Global state (in production, use Redis/database)
failed_attempts = {}
blocked_ips = {}
rate_limit_store = {}
users_db = {}

# Pure functions for security operations

def hash_password(password: str, salt: str) -> str:
    """Hash password using PBKDF2"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt.encode(),
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key.decode()

def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against hash"""
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        kdf.verify(password.encode(), base64.urlsafe_b64decode(hashed))
        return True
    except Exception:
        return False

def create_jwt_token(data: Dict[str, Any], secret_key: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, secret_key, algorithm="HS256")

def verify_jwt_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        return jwt.decode(token, secret_key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def encrypt_data(data: str, key: str) -> str:
    """Encrypt data using Fernet"""
    cipher = Fernet(base64.urlsafe_b64encode(key.encode()))
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: str) -> str:
    """Decrypt data using Fernet"""
    cipher = Fernet(base64.urlsafe_b64encode(key.encode()))
    return cipher.decrypt(encrypted_data.encode()).decode()

def validate_email(email: str) -> bool:
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength"""
    import re
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

def sanitize_input(input_str: str) -> str:
    """Sanitize user input"""
    import re
    sanitized = re.sub(r'[<>"\']', '', input_str)
    sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
    return sanitized.strip()

def validate_url(url: str) -> bool:
    """Validate URL format and security"""
    import re
    if not url.startswith(('http://', 'https://')):
        return False
    
    malicious_patterns = [
        r'javascript:', r'data:', r'vbscript:', r'file:', r'ftp:'
    ]
    
    return not any(re.search(pattern, url, re.IGNORECASE) for pattern in malicious_patterns)

def is_ip_blocked(client_ip: str) -> bool:
    """Check if IP is blocked"""
    if client_ip not in blocked_ips:
        return False
    
    if time.time() - blocked_ips[client_ip] > SECURITY_CONFIG["lockout_duration"]:
        del blocked_ips[client_ip]
        if client_ip in failed_attempts:
            del failed_attempts[client_ip]
        return False
    
    return True

def check_login_attempt(client_ip: str, success: bool) -> bool:
    """Check login attempt and apply rate limiting"""
    if success:
        if client_ip in failed_attempts:
            del failed_attempts[client_ip]
        return True
    
    if client_ip not in failed_attempts:
        failed_attempts[client_ip] = 1
    else:
        failed_attempts[client_ip] += 1
    
    if failed_attempts[client_ip] >= SECURITY_CONFIG["max_login_attempts"]:
        blocked_ips[client_ip] = time.time()
        return False
    
    return True

def is_rate_limit_exceeded(client_ip: str) -> bool:
    """Check rate limiting"""
    now = time.time()
    
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    
    # Remove old requests
    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip]
        if now - req_time < SECURITY_CONFIG["rate_limit_window"]
    ]
    
    if len(rate_limit_store[client_ip]) >= SECURITY_CONFIG["rate_limit_requests"]:
        return True
    
    rate_limit_store[client_ip].append(now)
    return False

def detect_suspicious_activity(request_data: str) -> List[str]:
    """Detect suspicious patterns"""
    import re
    suspicious_patterns = [
        r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
        r'(<script|javascript:|vbscript:)',
        r'(\.\./|\.\.\\)',
        r'(union.*select|select.*union)',
        r'(exec\(|eval\(|system\()',
    ]
    
    detected = []
    for pattern in suspicious_patterns:
        if re.search(pattern, request_data, re.IGNORECASE):
            detected.append(pattern)
    
    return detected

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security event"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    }
    print(f"SECURITY_LOG: {json.dumps(log_entry)}")

def log_access(user: str, resource: str, action: str, success: bool, ip: str) -> None:
    """Log access attempt"""
    log_security_event("ACCESS", {
        "user": user,
        "resource": resource,
        "action": action,
        "success": success,
        "ip": ip
    })

# Higher-order functions for security decorators

def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract token from kwargs or request
        token = kwargs.get('token')
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            payload = verify_jwt_token(token, SECURITY_CONFIG["secret_key"])
            kwargs['current_user'] = payload
            return await func(*args, **kwargs)
        except HTTPException:
            raise HTTPException(status_code=401, detail="Invalid authentication")
    
    return wrapper

def require_permission(permission: str) -> Callable:
    """Decorator to require specific permission"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user', {})
            user_permissions = user.get('permissions', [])
            
            if permission not in user_permissions:
                raise HTTPException(status_code=403, detail="Permission denied")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_requests: int, window_seconds: int) -> Callable:
    """Decorator for rate limiting"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                return await func(*args, **kwargs)
            
            client_ip = request.client.host
            
            # Check rate limit
            now = time.time()
            if client_ip not in rate_limit_store:
                rate_limit_store[client_ip] = []
            
            rate_limit_store[client_ip] = [
                req_time for req_time in rate_limit_store[client_ip]
                if now - req_time < window_seconds
            ]
            
            if len(rate_limit_store[client_ip]) >= max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            rate_limit_store[client_ip].append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(validation_func: Callable) -> Callable:
    """Decorator for input validation"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Apply validation function to input data
            if 'data' in kwargs:
                kwargs['data'] = validation_func(kwargs['data'])
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Functional security operations

def register_user(email: str, password: str) -> Dict[str, Any]:
    """Register user with security validation"""
    # Validate email
    if not validate_email(email):
        raise ValueError("Invalid email format")
    
    # Check if user exists
    if email in users_db:
        raise ValueError("User already exists")
    
    # Validate password strength
    password_validation = validate_password_strength(password)
    if not password_validation["valid"]:
        raise ValueError(f"Password validation failed: {password_validation['errors']}")
    
    # Hash password
    hashed_password = hash_password(password, SECURITY_CONFIG["salt"])
    
    # Store user
    users_db[email] = {
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "permissions": ["user"]
    }
    
    log_access(email, "/auth/register", "register", True, "127.0.0.1")
    
    return {
        "success": True,
        "message": "User registered successfully",
        "data": {"email": email}
    }

def authenticate_user(email: str, password: str, client_ip: str) -> Optional[str]:
    """Authenticate user with security checks"""
    # Check IP blocking
    if is_ip_blocked(client_ip):
        raise HTTPException(status_code=429, detail="IP blocked")
    
    # Check rate limiting
    if is_rate_limit_exceeded(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Verify user exists
    if email not in users_db:
        check_login_attempt(client_ip, False)
        log_access("unknown", "/auth/login", "login", False, client_ip)
        return None
    
    # Verify password
    user = users_db[email]
    if not verify_password(password, user["hashed_password"], SECURITY_CONFIG["salt"]):
        check_login_attempt(client_ip, False)
        log_access(email, "/auth/login", "login", False, client_ip)
        return None
    
    # Successful authentication
    check_login_attempt(client_ip, True)
    log_access(email, "/auth/login", "login", True, client_ip)
    
    # Generate token
    token = create_jwt_token(
        data={"sub": email, "permissions": user["permissions"]},
        secret_key=SECURITY_CONFIG["secret_key"],
        expires_delta=timedelta(minutes=SECURITY_CONFIG["jwt_expire_minutes"])
    )
    
    return token

def process_video_secure(video_data: Dict[str, Any], user: Dict[str, Any], client_ip: str) -> Dict[str, Any]:
    """Process video with security validation"""
    # Monitor for suspicious activity
    suspicious_patterns = detect_suspicious_activity(str(video_data))
    if suspicious_patterns:
        log_security_event("SUSPICIOUS_ACTIVITY", {
            "patterns": suspicious_patterns,
            "user": user.get("sub"),
            "ip": client_ip
        })
        raise HTTPException(status_code=400, detail="Suspicious activity detected")
    
    # Validate and sanitize input
    title = sanitize_input(video_data.get("title", ""))
    if len(title) > 100:
        raise ValueError("Title too long")
    
    url = video_data.get("url", "")
    if not validate_url(url):
        raise ValueError("Invalid or malicious URL")
    
    description = sanitize_input(video_data.get("description", ""))
    if len(description) > 1000:
        raise ValueError("Description too long")
    
    # Encrypt sensitive data
    encrypted_description = encrypt_data(description, SECURITY_CONFIG["encryption_key"])
    
    return {
        "id": secrets.token_urlsafe(16),
        "title": title,
        "url": url,
        "encrypted_description": encrypted_description,
        "user": user.get("sub"),
        "processed_at": datetime.utcnow()
    }

# Functional API endpoints (example usage)

@require_authentication
@rate_limit(max_requests=50, window_seconds=60)
async def create_video_endpoint(
    video_data: Dict[str, Any],
    current_user: Dict[str, Any],
    request: Request
) -> Dict[str, Any]:
    """Create video with security"""
    client_ip = request.client.host
    
    result = process_video_secure(video_data, current_user, client_ip)
    
    log_access(
        current_user["sub"], 
        "/videos", 
        "create", 
        True, 
        client_ip
    )
    
    return {
        "success": True,
        "message": "Video created successfully",
        "data": result
    }

@require_authentication
@require_permission("admin")
async def admin_endpoint(current_user: Dict[str, Any]) -> Dict[str, Any]:
    """Admin-only endpoint"""
    return {
        "success": True,
        "message": "Admin access granted",
        "data": {"user": current_user["sub"]}
    }

# Utility functions

def get_security_metrics() -> Dict[str, Any]:
    """Get security metrics"""
    return {
        "blocked_ips": len(blocked_ips),
        "failed_attempts": len(failed_attempts),
        "rate_limited_ips": len(rate_limit_store),
        "total_users": len(users_db)
    }

def clear_security_data() -> None:
    """Clear security data (for testing)"""
    global failed_attempts, blocked_ips, rate_limit_store
    failed_attempts.clear()
    blocked_ips.clear()
    rate_limit_store.clear()

# Example usage
async def main():
    """Example usage of functional security"""
    print("🔒 Functional Security Example")
    
    # Register user
    try:
        register_user("user@example.com", "SecurePass123!")
        print("✅ User registered")
    except ValueError as e:
        print(f"❌ Registration failed: {e}")
    
    # Authenticate user
    token = authenticate_user("user@example.com", "SecurePass123!", "127.0.0.1")
    if token:
        print(f"✅ User authenticated, token: {token[:20]}...")
    
    # Process video
    video_data = {
        "title": "My Video",
        "url": "https://example.com/video.mp4",
        "description": "A great video"
    }
    
    user_data = {"sub": "user@example.com", "permissions": ["user"]}
    result = process_video_secure(video_data, user_data, "127.0.0.1")
    print(f"✅ Video processed: {result['id']}")
    
    # Get metrics
    metrics = get_security_metrics()
    print(f"📊 Security metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main()) 