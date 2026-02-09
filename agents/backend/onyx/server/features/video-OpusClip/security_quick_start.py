#!/usr/bin/env python3
"""
Security Quick Start for Video-OpusClip
Concise, technical examples for immediate implementation
"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel, validator

# Import security components
from security_implementation import (
    SecurityConfig, InputValidator, PasswordManager, DataEncryption,
    JWTManager, RateLimiter, IntrusionDetector, SecurityLogger
)

# Initialize security
config = SecurityConfig()
password_mgr = PasswordManager(config.salt)
encryption = DataEncryption(config.encryption_key)
jwt_mgr = JWTManager(config.secret_key)
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
intrusion_detector = IntrusionDetector(max_failed_attempts=5, lockout_duration=900)
security_logger = SecurityLogger()

# Security models
class LoginRequest(BaseModel):
    email: str
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email')
        return v.lower()

class VideoRequest(BaseModel):
    title: str
    url: str
    
    @validator('title')
    def validate_title(cls, v):
        sanitized = InputValidator.sanitize_input(v)
        if len(sanitized) > 100:
            raise ValueError('Title too long')
        return sanitized
    
    @validator('url')
    def validate_url(cls, v):
        if not InputValidator.validate_url(v):
            raise ValueError('Invalid URL')
        return v

# Security middleware
async def security_middleware(request, call_next):
    """Basic security middleware"""
    client_ip = request.client.host
    
    # Check IP blocking
    if intrusion_detector.is_ip_blocked(client_ip):
        raise HTTPException(status_code=429, detail="IP blocked")
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    
    # Security headers
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block"
    })
    
    return response

# Authentication dependency
security = HTTPBearer()

async def get_current_user(credentials = Depends(security)):
    """Get authenticated user"""
    try:
        payload = jwt_mgr.verify_token(credentials.credentials)
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Mock user database
users_db = {}

# Security endpoints
async def register_user(email: str, password: str) -> bool:
    """Register user securely"""
    if email in users_db:
        return False
    
    # Validate password
    validation = InputValidator.validate_password_strength(password)
    if not validation['valid']:
        raise ValueError(f"Password weak: {validation['errors']}")
    
    # Hash password
    hashed = password_mgr.hash_password(password)
    users_db[email] = {"hashed_password": hashed}
    
    security_logger.log_access(email, "/register", "register", True, "127.0.0.1")
    return True

async def login_user(email: str, password: str, client_ip: str) -> Optional[str]:
    """Login user with security checks"""
    # IP blocking check
    if intrusion_detector.is_ip_blocked(client_ip):
        raise HTTPException(status_code=429, detail="IP blocked")
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Verify user
    if email not in users_db:
        intrusion_detector.check_login_attempt(client_ip, False)
        security_logger.log_access("unknown", "/login", "login", False, client_ip)
        return None
    
    # Verify password
    user = users_db[email]
    if not password_mgr.verify_password(password, user["hashed_password"]):
        intrusion_detector.check_login_attempt(client_ip, False)
        security_logger.log_access(email, "/login", "login", False, client_ip)
        return None
    
    # Success
    intrusion_detector.check_login_attempt(client_ip, True)
    security_logger.log_access(email, "/login", "login", True, client_ip)
    
    # Generate token
    token = jwt_mgr.create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=30)
    )
    
    return token

async def process_video_secure(video_data: VideoRequest, user: Dict) -> Dict:
    """Process video with security validation"""
    # Monitor for suspicious activity
    suspicious = intrusion_detector.detect_suspicious_activity(str(video_data.dict()))
    if suspicious:
        security_logger.log_security_event("SUSPICIOUS_ACTIVITY", {
            "patterns": suspicious,
            "user": user.get("sub")
        })
        raise HTTPException(status_code=400, detail="Suspicious activity")
    
    # Encrypt sensitive data
    encrypted_title = encryption.encrypt(video_data.title)
    
    return {
        "id": secrets.token_urlsafe(16),
        "encrypted_title": encrypted_title,
        "url": video_data.url,
        "user": user.get("sub"),
        "processed_at": datetime.utcnow()
    }

# Quick security tests
def run_security_tests():
    """Run basic security tests"""
    print("🔒 Running Security Tests...")
    
    # Test password hashing
    password = "SecurePass123!"
    hashed = password_mgr.hash_password(password)
    assert password_mgr.verify_password(password, hashed)
    print("✅ Password hashing works")
    
    # Test encryption
    data = "secret data"
    encrypted = encryption.encrypt(data)
    decrypted = encryption.decrypt(encrypted)
    assert decrypted == data
    print("✅ Encryption works")
    
    # Test JWT
    token = jwt_mgr.create_access_token({"sub": "test@example.com"})
    payload = jwt_mgr.verify_token(token)
    assert payload["sub"] == "test@example.com"
    print("✅ JWT works")
    
    # Test rate limiting
    assert rate_limiter.is_allowed("test_ip")
    print("✅ Rate limiting works")
    
    # Test input validation
    assert InputValidator.validate_email("test@example.com")
    assert not InputValidator.validate_email("invalid-email")
    print("✅ Input validation works")
    
    print("🎉 All security tests passed!")

# Example usage
async def main():
    """Main security example"""
    print("🚀 Security Quick Start Example")
    
    # Run tests
    run_security_tests()
    
    # Register user
    await register_user("user@example.com", "SecurePass123!")
    print("✅ User registered")
    
    # Login user
    token = await login_user("user@example.com", "SecurePass123!", "127.0.0.1")
    print(f"✅ User logged in, token: {token[:20]}...")
    
    # Process video
    video_data = VideoRequest(
        title="My Video",
        url="https://example.com/video.mp4"
    )
    
    user_data = {"sub": "user@example.com"}
    result = await process_video_secure(video_data, user_data)
    print(f"✅ Video processed: {result['id']}")
    
    print("🎯 Security implementation complete!")

if __name__ == "__main__":
    asyncio.run(main()) 