#!/usr/bin/env python3
"""
Security Examples for Video-OpusClip
Concise, technical examples with accurate Python code
"""

import asyncio
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import HTTPException, Depends
from pydantic import BaseModel, validator

# Import security components
from security_implementation import (
    SecurityConfig, InputValidator, PasswordManager, DataEncryption,
    JWTManager, RateLimiter, IntrusionDetector, SecurityLogger,
    IncidentResponse, SecurityIncident, IncidentType, SecurityLevel
)

# Initialize security components
config = SecurityConfig()
password_mgr = PasswordManager(config.salt)
encryption = DataEncryption(config.encryption_key)
jwt_mgr = JWTManager(config.secret_key)
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
intrusion_detector = IntrusionDetector(max_failed_attempts=5, lockout_duration=900)
security_logger = SecurityLogger()
incident_response = IncidentResponse()

# Example 1: Secure User Authentication
class SecureUserAuth:
    """Secure user authentication example"""
    
    def __init__(self):
        self.users = {}  # In production, use database
    
    def register_user(self, email: str, password: str) -> bool:
        """Register user with secure password hashing"""
        if email in self.users:
            return False
        
        # Validate password strength
        validation = InputValidator.validate_password_strength(password)
        if not validation['valid']:
            raise ValueError(f"Password too weak: {validation['errors']}")
        
        # Hash password
        hashed_password = password_mgr.hash_password(password)
        
        # Store user
        self.users[email] = {
            'email': email,
            'hashed_password': hashed_password,
            'created_at': datetime.utcnow()
        }
        
        security_logger.log_access(email, "/auth/register", "register", True, "127.0.0.1")
        return True
    
    def authenticate_user(self, email: str, password: str, client_ip: str) -> Optional[str]:
        """Authenticate user with security checks"""
        # Check IP blocking
        if intrusion_detector.is_ip_blocked(client_ip):
            raise HTTPException(status_code=429, detail="IP blocked")
        
        # Check rate limiting
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Verify user exists
        if email not in self.users:
            intrusion_detector.check_login_attempt(client_ip, False)
            security_logger.log_access("unknown", "/auth/login", "login", False, client_ip)
            return None
        
        # Verify password
        user = self.users[email]
        if not password_mgr.verify_password(password, user['hashed_password']):
            intrusion_detector.check_login_attempt(client_ip, False)
            security_logger.log_access(email, "/auth/login", "login", False, client_ip)
            return None
        
        # Successful authentication
        intrusion_detector.check_login_attempt(client_ip, True)
        security_logger.log_access(email, "/auth/login", "login", True, client_ip)
        
        # Generate JWT token
        token = jwt_mgr.create_access_token(
            data={"sub": email, "role": "user"},
            expires_delta=timedelta(minutes=30)
        )
        
        return token

# Example 2: Secure Input Validation
class SecureVideoProcessor:
    """Secure video processing with input validation"""
    
    def process_video_request(self, data: Dict[str, any]) -> Dict[str, any]:
        """Process video request with security validation"""
        
        # Validate and sanitize title
        title = InputValidator.sanitize_input(data.get('title', ''))
        if len(title) > 100:
            raise ValueError("Title too long")
        
        # Validate URL
        url = data.get('url', '')
        if not InputValidator.validate_url(url):
            raise ValueError("Invalid or malicious URL")
        
        # Validate description
        description = InputValidator.sanitize_input(data.get('description', ''))
        if len(description) > 1000:
            raise ValueError("Description too long")
        
        # Validate tags
        tags = []
        for tag in data.get('tags', [])[:10]:  # Limit to 10 tags
            sanitized_tag = InputValidator.sanitize_input(tag)
            if sanitized_tag and len(sanitized_tag) <= 50:
                tags.append(sanitized_tag)
        
        return {
            'title': title,
            'url': url,
            'description': description,
            'tags': tags,
            'processed_at': datetime.utcnow()
        }

# Example 3: Data Encryption
class SecureDataManager:
    """Secure data management with encryption"""
    
    def store_sensitive_data(self, data: str) -> str:
        """Store sensitive data with encryption"""
        encrypted_data = encryption.encrypt(data)
        return encrypted_data
    
    def retrieve_sensitive_data(self, encrypted_data: str) -> str:
        """Retrieve and decrypt sensitive data"""
        decrypted_data = encryption.decrypt(encrypted_data)
        return decrypted_data

# Example 4: Security Monitoring
class SecurityMonitor:
    """Security monitoring and incident response"""
    
    def monitor_request(self, request_data: str, client_ip: str, user_id: str):
        """Monitor request for suspicious activity"""
        
        # Detect suspicious patterns
        suspicious_patterns = intrusion_detector.detect_suspicious_activity(request_data)
        
        if suspicious_patterns:
            # Create security incident
            incident = SecurityIncident(
                id=secrets.token_urlsafe(16),
                type=IncidentType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.MEDIUM,
                description=f"Suspicious patterns detected: {suspicious_patterns}",
                timestamp=datetime.utcnow(),
                source_ip=client_ip,
                user_id=user_id,
                details={"patterns": suspicious_patterns}
            )
            
            incident_response.create_incident(incident)
            return False
        
        return True

# Example 5: Secure API Endpoint
async def secure_video_endpoint(
    video_data: Dict[str, any],
    client_ip: str,
    user_token: str
) -> Dict[str, any]:
    """Secure video processing endpoint"""
    
    try:
        # Verify JWT token
        payload = jwt_mgr.verify_token(user_token)
        user_email = payload.get("sub")
        
        # Monitor for suspicious activity
        monitor = SecurityMonitor()
        if not monitor.monitor_request(str(video_data), client_ip, user_email):
            raise HTTPException(status_code=400, detail="Suspicious activity detected")
        
        # Process video securely
        processor = SecureVideoProcessor()
        processed_data = processor.process_video_request(video_data)
        
        # Encrypt sensitive data
        data_manager = SecureDataManager()
        encrypted_description = data_manager.store_sensitive_data(processed_data['description'])
        
        return {
            "success": True,
            "data": {
                "id": secrets.token_urlsafe(16),
                "title": processed_data['title'],
                "url": processed_data['url'],
                "encrypted_description": encrypted_description,
                "tags": processed_data['tags']
            }
        }
        
    except Exception as e:
        security_logger.log_security_event("API_ERROR", {
            "error": str(e),
            "ip": client_ip,
            "user": user_email if 'user_email' in locals() else "unknown"
        })
        raise HTTPException(status_code=500, detail="Processing failed")

# Example 6: Rate Limiting Implementation
class APIRateLimiter:
    """API rate limiting with security"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
        self.user_limits = {}  # Per-user rate limits
    
    def check_rate_limit(self, identifier: str, client_ip: str) -> bool:
        """Check rate limit for user/IP"""
        
        # Global rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            return False
        
        # Per-user rate limiting
        if identifier not in self.user_limits:
            self.user_limits[identifier] = RateLimiter(max_requests=20, window_seconds=60)
        
        return self.user_limits[identifier].is_allowed(identifier)

# Example 7: Security Headers Middleware
def add_security_headers(response):
    """Add security headers to response"""
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    })
    return response

# Example 8: Secure Database Operations
class SecureDatabaseOps:
    """Secure database operations with parameterized queries"""
    
    async def get_video_safe(self, video_id: str, user_id: str) -> Optional[Dict]:
        """Get video with access control"""
        
        # Parameterized query to prevent SQL injection
        query = "SELECT * FROM videos WHERE id = $1 AND user_id = $2"
        # In real implementation: result = await db.fetch_one(query, video_id, user_id)
        
        # Mock result for example
        if video_id and user_id:
            return {
                "id": video_id,
                "title": "Sample Video",
                "user_id": user_id
            }
        return None
    
    async def create_video_safe(self, video_data: Dict, user_id: str) -> str:
        """Create video with validation"""
        
        # Validate input
        processor = SecureVideoProcessor()
        processed_data = processor.process_video_request(video_data)
        
        # Parameterized insert
        query = """
            INSERT INTO videos (title, url, description, user_id, created_at)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
        """
        # In real implementation: result = await db.fetch_one(query, ...)
        
        return secrets.token_urlsafe(16)

# Example 9: Security Testing
def test_security_components():
    """Test security components"""
    
    # Test password hashing
    password = "SecurePass123!"
    hashed = password_mgr.hash_password(password)
    assert password_mgr.verify_password(password, hashed)
    assert not password_mgr.verify_password("wrong", hashed)
    
    # Test encryption
    data = "sensitive information"
    encrypted = encryption.encrypt(data)
    decrypted = encryption.decrypt(encrypted)
    assert decrypted == data
    
    # Test JWT
    token = jwt_mgr.create_access_token({"sub": "test@example.com"})
    payload = jwt_mgr.verify_token(token)
    assert payload["sub"] == "test@example.com"
    
    # Test rate limiting
    assert rate_limiter.is_allowed("test_ip")
    
    # Test intrusion detection
    assert not intrusion_detector.is_ip_blocked("test_ip")
    
    print("All security tests passed!")

# Example 10: Security Configuration
class SecuritySettings:
    """Security configuration management"""
    
    def __init__(self):
        self.settings = {
            "max_login_attempts": 5,
            "lockout_duration": 900,
            "rate_limit_requests": 100,
            "rate_limit_window": 60,
            "jwt_expire_minutes": 30,
            "password_min_length": 8,
            "require_special_chars": True,
            "session_timeout": 3600
        }
    
    def get_setting(self, key: str) -> any:
        """Get security setting"""
        return self.settings.get(key)
    
    def update_setting(self, key: str, value: any):
        """Update security setting"""
        self.settings[key] = value

# Usage examples
if __name__ == "__main__":
    # Test security components
    test_security_components()
    
    # Example usage
    auth = SecureUserAuth()
    auth.register_user("user@example.com", "SecurePass123!")
    
    token = auth.authenticate_user("user@example.com", "SecurePass123!", "127.0.0.1")
    print(f"Authentication token: {token}")
    
    # Process video securely
    processor = SecureVideoProcessor()
    video_data = {
        "title": "My Video",
        "url": "https://example.com/video.mp4",
        "description": "A great video",
        "tags": ["fun", "educational"]
    }
    
    processed = processor.process_video_request(video_data)
    print(f"Processed video: {processed}") 