#!/usr/bin/env python3
"""
Functional Security Examples for Video-OpusClip
Declarative programming with pure functions
"""

import asyncio
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from functools import wraps, partial, reduce

from fastapi import HTTPException
from pydantic import BaseModel, validator

# Pure security functions

def hash_password(password: str, salt: str) -> str:
    """Hash password using PBKDF2"""
    import hashlib
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against hash"""
    return hash_password(password, salt) == hashed

def create_jwt_token(data: Dict, secret: str, expires_minutes: int = 30) -> str:
    """Create JWT token"""
    import jwt
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=expires_minutes)}
    return jwt.encode(payload, secret, algorithm="HS256")

def verify_jwt_token(token: str, secret: str) -> Dict:
    """Verify JWT token"""
    import jwt
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def encrypt_data(data: str, key: str) -> str:
    """Encrypt data"""
    import base64
    from cryptography.fernet import Fernet
    cipher = Fernet(base64.urlsafe_b64encode(key.encode()))
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted: str, key: str) -> str:
    """Decrypt data"""
    import base64
    from cryptography.fernet import Fernet
    cipher = Fernet(base64.urlsafe_b64encode(key.encode()))
    return cipher.decrypt(encrypted.encode()).decode()

# Input validation functions

def validate_email(email: str) -> bool:
    """Validate email"""
    import re
    return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', email))

def validate_password(password: str) -> Dict[str, any]:
    """Validate password strength"""
    import re
    checks = [
        (len(password) >= 8, "Too short"),
        (re.search(r'[A-Z]', password), "No uppercase"),
        (re.search(r'[a-z]', password), "No lowercase"),
        (re.search(r'\d', password), "No digit"),
        (re.search(r'[!@#$%^&*]', password), "No special char")
    ]
    errors = [msg for check, msg in checks if not check]
    return {"valid": not errors, "errors": errors}

def sanitize_input(text: str) -> str:
    """Sanitize input"""
    import re
    return re.sub(r'[<>"\']', '', text).strip()

def validate_url(url: str) -> bool:
    """Validate URL"""
    return url.startswith(('http://', 'https://')) and 'javascript:' not in url.lower()

# Security state management (functional approach)

def create_security_state() -> Dict[str, any]:
    """Create initial security state"""
    return {
        "failed_attempts": {},
        "blocked_ips": {},
        "rate_limits": {},
        "users": {}
    }

def update_failed_attempts(state: Dict, ip: str, success: bool) -> Dict:
    """Update failed attempts"""
    if success:
        return {**state, "failed_attempts": {k: v for k, v in state["failed_attempts"].items() if k != ip}}
    
    attempts = state["failed_attempts"].get(ip, 0) + 1
    new_failed = {**state["failed_attempts"], ip: attempts}
    
    if attempts >= 5:
        new_blocked = {**state["blocked_ips"], ip: time.time()}
        return {**state, "failed_attempts": new_failed, "blocked_ips": new_blocked}
    
    return {**state, "failed_attempts": new_failed}

def is_ip_blocked(state: Dict, ip: str) -> bool:
    """Check if IP is blocked"""
    if ip not in state["blocked_ips"]:
        return False
    
    blocked_time = state["blocked_ips"][ip]
    if time.time() - blocked_time > 900:  # 15 minutes
        return False
    
    return True

def check_rate_limit(state: Dict, ip: str, max_requests: int = 100, window: int = 60) -> tuple[bool, Dict]:
    """Check rate limit"""
    now = time.time()
    requests = state["rate_limits"].get(ip, [])
    
    # Remove old requests
    valid_requests = [req for req in requests if now - req < window]
    
    if len(valid_requests) >= max_requests:
        return False, state
    
    new_requests = valid_requests + [now]
    new_state = {**state, "rate_limits": {**state["rate_limits"], ip: new_requests}}
    
    return True, new_state

# Higher-order functions for security

def with_authentication(func: Callable) -> Callable:
    """Authentication decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token = kwargs.get('token')
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            user = verify_jwt_token(token, "secret-key")
            kwargs['user'] = user
            return await func(*args, **kwargs)
        except:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    return wrapper

def with_rate_limit(max_requests: int, window: int) -> Callable:
    """Rate limiting decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = kwargs.get('security_state', create_security_state())
            ip = kwargs.get('client_ip', 'unknown')
            
            allowed, new_state = check_rate_limit(state, ip, max_requests, window)
            if not allowed:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            kwargs['security_state'] = new_state
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def with_input_validation(validation_func: Callable) -> Callable:
    """Input validation decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if 'data' in kwargs:
                kwargs['data'] = validation_func(kwargs['data'])
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Functional security operations

def register_user(email: str, password: str, state: Dict) -> tuple[Dict, Dict]:
    """Register user functionally"""
    # Validate input
    if not validate_email(email):
        raise ValueError("Invalid email")
    
    if email in state["users"]:
        raise ValueError("User exists")
    
    password_check = validate_password(password)
    if not password_check["valid"]:
        raise ValueError(f"Password invalid: {password_check['errors']}")
    
    # Create user
    hashed = hash_password(password, "salt")
    new_user = {
        "email": email,
        "hashed_password": hashed,
        "created_at": datetime.utcnow()
    }
    
    new_state = {**state, "users": {**state["users"], email: new_user}}
    
    return {"success": True, "user": new_user}, new_state

def authenticate_user(email: str, password: str, client_ip: str, state: Dict) -> tuple[Optional[str], Dict]:
    """Authenticate user functionally"""
    # Check IP blocking
    if is_ip_blocked(state, client_ip):
        raise HTTPException(status_code=429, detail="IP blocked")
    
    # Check rate limit
    allowed, state = check_rate_limit(state, client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Verify user
    if email not in state["users"]:
        state = update_failed_attempts(state, client_ip, False)
        return None, state
    
    user = state["users"][email]
    if not verify_password(password, user["hashed_password"], "salt"):
        state = update_failed_attempts(state, client_ip, False)
        return None, state
    
    # Success
    state = update_failed_attempts(state, client_ip, True)
    token = create_jwt_token({"sub": email}, "secret-key")
    
    return token, state

def process_video_secure(video_data: Dict, user: Dict, client_ip: str) -> Dict:
    """Process video with security validation"""
    # Validate input
    title = sanitize_input(video_data.get("title", ""))
    if len(title) > 100:
        raise ValueError("Title too long")
    
    url = video_data.get("url", "")
    if not validate_url(url):
        raise ValueError("Invalid URL")
    
    description = sanitize_input(video_data.get("description", ""))
    if len(description) > 1000:
        raise ValueError("Description too long")
    
    # Encrypt sensitive data
    encrypted_desc = encrypt_data(description, "encryption-key")
    
    return {
        "id": secrets.token_urlsafe(16),
        "title": title,
        "url": url,
        "encrypted_description": encrypted_desc,
        "user": user.get("sub"),
        "processed_at": datetime.utcnow()
    }

# Functional composition examples

def compose(*functions):
    """Function composition"""
    return lambda x: reduce(lambda acc, f: f(acc), reversed(functions), x)

def pipeline(data, *functions):
    """Pipeline data through functions"""
    return compose(*functions)(data)

# Example: Data processing pipeline
def validate_video_data(data: Dict) -> Dict:
    """Validate video data"""
    return {
        **data,
        "title": sanitize_input(data.get("title", "")),
        "url": data.get("url") if validate_url(data.get("url", "")) else None
    }

def encrypt_sensitive_data(data: Dict) -> Dict:
    """Encrypt sensitive data"""
    return {
        **data,
        "description": encrypt_data(data.get("description", ""), "key")
    }

def add_metadata(data: Dict, user: str) -> Dict:
    """Add metadata"""
    return {
        **data,
        "id": secrets.token_urlsafe(16),
        "user": user,
        "processed_at": datetime.utcnow()
    }

# Complete pipeline
video_processing_pipeline = compose(
    validate_video_data,
    encrypt_sensitive_data,
    lambda data: add_metadata(data, "user@example.com")
)

# Security utilities

def get_security_metrics(state: Dict) -> Dict:
    """Get security metrics"""
    return {
        "blocked_ips": len(state["blocked_ips"]),
        "failed_attempts": len(state["failed_attempts"]),
        "rate_limited_ips": len(state["rate_limits"]),
        "total_users": len(state["users"])
    }

def log_security_event(event_type: str, details: Dict) -> None:
    """Log security event"""
    print(f"SECURITY: {event_type} - {details}")

# Example usage
async def main():
    """Example usage of functional security"""
    print("🔒 Functional Security Examples")
    
    # Initialize state
    state = create_security_state()
    
    # Register user
    try:
        user_data, state = register_user("user@example.com", "SecurePass123!", state)
        print("✅ User registered")
    except ValueError as e:
        print(f"❌ Registration failed: {e}")
    
    # Authenticate user
    token, state = authenticate_user("user@example.com", "SecurePass123!", "127.0.0.1", state)
    if token:
        print(f"✅ User authenticated, token: {token[:20]}...")
    
    # Process video using pipeline
    video_data = {
        "title": "My Video<script>alert('xss')</script>",
        "url": "https://example.com/video.mp4",
        "description": "A great video"
    }
    
    processed_video = video_processing_pipeline(video_data)
    print(f"✅ Video processed: {processed_video['id']}")
    print(f"   Title: {processed_video['title']}")
    print(f"   Encrypted description: {processed_video['description'][:20]}...")
    
    # Get metrics
    metrics = get_security_metrics(state)
    print(f"📊 Security metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main()) 