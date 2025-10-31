#!/usr/bin/env python3
"""
Descriptive Security Examples for Video-OpusClip
Examples using descriptive variable names with auxiliary verbs
"""

import asyncio
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from functools import wraps

from fastapi import HTTPException
from pydantic import BaseModel, validator

# Descriptive security functions with auxiliary verbs

def is_password_strong_enough(password_string: str) -> Dict[str, Any]:
    """Check if password meets strength requirements"""
    import re
    
    validation_checks = [
        (len(password_string) >= 8, "Password is too short"),
        (re.search(r'[A-Z]', password_string), "Password lacks uppercase letter"),
        (re.search(r'[a-z]', password_string), "Password lacks lowercase letter"),
        (re.search(r'\d', password_string), "Password lacks numeric digit"),
        (re.search(r'[!@#$%^&*]', password_string), "Password lacks special character")
    ]
    
    failed_checks = [message for check_passed, message in validation_checks if not check_passed]
    
    return {
        "is_password_valid": len(failed_checks) == 0,
        "validation_errors": failed_checks,
        "password_strength_score": max(0, 10 - len(failed_checks) * 2)
    }

def is_email_address_valid(email_address: str) -> bool:
    """Check if email address has valid format"""
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email_address))

def is_url_safe_for_processing(url_string: str) -> bool:
    """Check if URL is safe and not malicious"""
    import re
    
    # Check if URL uses secure protocols
    has_secure_protocol = url_string.startswith(('http://', 'https://'))
    if not has_secure_protocol:
        return False
    
    # Check for malicious patterns
    malicious_patterns = ['javascript:', 'data:', 'vbscript:', 'file:', 'ftp:']
    for malicious_pattern in malicious_patterns:
        if malicious_pattern in url_string.lower():
            return False
    
    return True

def has_suspicious_content(input_text: str) -> List[str]:
    """Check if input contains suspicious patterns"""
    import re
    
    suspicious_patterns = [
        r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
        r'(<script|javascript:|vbscript:)',
        r'(\.\./|\.\.\\)',
        r'(union.*select|select.*union)',
        r'(exec\(|eval\(|system\()',
    ]
    
    detected_patterns = []
    for pattern in suspicious_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            detected_patterns.append(pattern)
    
    return detected_patterns

def is_user_authenticated(token_string: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """Check if user is authenticated with valid token"""
    import jwt
    
    try:
        decoded_token = jwt.decode(token_string, secret_key, algorithms=["HS256"])
        return decoded_token
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

def is_ip_address_blocked(client_ip: str, blocked_ips: Dict[str, float], lockout_duration: int) -> bool:
    """Check if IP address is currently blocked"""
    if client_ip not in blocked_ips:
        return False
    
    current_time = time.time()
    block_timestamp = blocked_ips[client_ip]
    
    if current_time - block_timestamp > lockout_duration:
        return False
    
    return True

def has_exceeded_rate_limit(client_ip: str, request_history: Dict[str, List[float]], max_requests: int, window_seconds: int) -> bool:
    """Check if client has exceeded rate limit"""
    current_time = time.time()
    
    if client_ip not in request_history:
        return False
    
    # Remove old requests
    recent_requests = [
        req_time for req_time in request_history[client_ip]
        if current_time - req_time < window_seconds
    ]
    
    return len(recent_requests) >= max_requests

def is_data_encrypted(data_string: str, encryption_key: str) -> str:
    """Encrypt data if not already encrypted"""
    from cryptography.fernet import Fernet
    
    try:
        # Try to decrypt to check if already encrypted
        cipher = Fernet(encryption_key.encode())
        cipher.decrypt(data_string.encode())
        return data_string  # Already encrypted
    except:
        # Not encrypted, encrypt it
        cipher = Fernet(encryption_key.encode())
        return cipher.encrypt(data_string.encode()).decode()

def is_input_sanitized(input_text: str) -> str:
    """Sanitize input to remove dangerous content"""
    import re
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', '"', "'"]
    for char in dangerous_chars:
        input_text = input_text.replace(char, '')
    
    # Remove script tags
    input_text = re.sub(r'<script.*?</script>', '', input_text, flags=re.IGNORECASE)
    
    return input_text.strip()

# Descriptive security classes

class DescriptiveSecurityValidator:
    """Security validator with descriptive method names"""
    
    def __init__(self):
        self.validation_rules = {
            "email": is_email_address_valid,
            "password": is_password_strong_enough,
            "url": is_url_safe_for_processing,
            "input": is_input_sanitized
        }
    
    def validate_user_registration_data(self, registration_data: Dict[str, str]) -> Dict[str, Any]:
        """Validate user registration data"""
        validation_results = {}
        
        # Validate email
        if "email" in registration_data:
            email_address = registration_data["email"]
            validation_results["email"] = {
                "is_valid": is_email_address_valid(email_address),
                "value": email_address
            }
        
        # Validate password
        if "password" in registration_data:
            password_string = registration_data["password"]
            validation_results["password"] = is_password_strong_enough(password_string)
        
        return validation_results
    
    def validate_video_upload_data(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video upload data"""
        validation_results = {}
        
        # Validate title
        if "title" in video_data:
            title_text = video_data["title"]
            sanitized_title = is_input_sanitized(title_text)
            validation_results["title"] = {
                "is_valid": len(sanitized_title) <= 100,
                "sanitized_value": sanitized_title,
                "original_value": title_text
            }
        
        # Validate URL
        if "url" in video_data:
            url_string = video_data["url"]
            validation_results["url"] = {
                "is_valid": is_url_safe_for_processing(url_string),
                "value": url_string
            }
        
        # Check for suspicious content
        if "description" in video_data:
            description_text = video_data["description"]
            suspicious_patterns = has_suspicious_content(description_text)
            validation_results["description"] = {
                "is_valid": len(suspicious_patterns) == 0,
                "suspicious_patterns": suspicious_patterns,
                "sanitized_value": is_input_sanitized(description_text)
            }
        
        return validation_results

class DescriptiveSecurityManager:
    """Security manager with descriptive method names"""
    
    def __init__(self, secret_key: str, encryption_key: str):
        self.secret_key = secret_key
        self.encryption_key = encryption_key
        self.failed_login_attempts = {}
        self.blocked_ip_addresses = {}
        self.request_history = {}
    
    def is_user_login_allowed(self, email_address: str, password_string: str, client_ip: str) -> Dict[str, Any]:
        """Check if user login is allowed"""
        # Check if IP is blocked
        if is_ip_address_blocked(client_ip, self.blocked_ip_addresses, 900):
            return {
                "is_login_allowed": False,
                "reason": "IP address is blocked",
                "client_ip": client_ip
            }
        
        # Check rate limiting
        if has_exceeded_rate_limit(client_ip, self.request_history, 100, 60):
            return {
                "is_login_allowed": False,
                "reason": "Rate limit exceeded",
                "client_ip": client_ip
            }
        
        # Validate credentials (mock implementation)
        is_email_valid = is_email_address_valid(email_address)
        is_password_valid = is_password_strong_enough(password_string)["is_password_valid"]
        
        if not is_email_valid or not is_password_valid:
            # Increment failed attempts
            if client_ip not in self.failed_login_attempts:
                self.failed_login_attempts[client_ip] = 1
            else:
                self.failed_login_attempts[client_ip] += 1
            
            # Block IP if too many failed attempts
            if self.failed_login_attempts[client_ip] >= 5:
                self.blocked_ip_addresses[client_ip] = time.time()
            
            return {
                "is_login_allowed": False,
                "reason": "Invalid credentials",
                "failed_attempts": self.failed_login_attempts.get(client_ip, 0)
            }
        
        # Successful login
        if client_ip in self.failed_login_attempts:
            del self.failed_login_attempts[client_ip]
        
        return {
            "is_login_allowed": True,
            "user_email": email_address,
            "client_ip": client_ip
        }
    
    def is_video_processing_safe(self, video_data: Dict[str, Any], user_token: str) -> Dict[str, Any]:
        """Check if video processing is safe"""
        # Validate user authentication
        user_data = is_user_authenticated(user_token, self.secret_key)
        if not user_data:
            return {
                "is_processing_safe": False,
                "reason": "Invalid authentication token"
            }
        
        # Validate video data
        validator = DescriptiveSecurityValidator()
        validation_results = validator.validate_video_upload_data(video_data)
        
        # Check if all validations passed
        all_validations_passed = all(
            result.get("is_valid", False) for result in validation_results.values()
        )
        
        if not all_validations_passed:
            return {
                "is_processing_safe": False,
                "reason": "Video data validation failed",
                "validation_errors": validation_results
            }
        
        # Check for suspicious content
        suspicious_content_found = any(
            "suspicious_patterns" in result and len(result["suspicious_patterns"]) > 0
            for result in validation_results.values()
        )
        
        if suspicious_content_found:
            return {
                "is_processing_safe": False,
                "reason": "Suspicious content detected",
                "validation_results": validation_results
            }
        
        return {
            "is_processing_safe": True,
            "user_data": user_data,
            "validation_results": validation_results
        }

# Descriptive security decorators

def require_valid_authentication(func: Callable) -> Callable:
    """Decorator to require valid authentication"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token_string = kwargs.get('token')
        secret_key = kwargs.get('secret_key', 'default-secret')
        
        user_data = is_user_authenticated(token_string, secret_key)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        kwargs['authenticated_user'] = user_data
        return await func(*args, **kwargs)
    
    return wrapper

def require_safe_input_data(func: Callable) -> Callable:
    """Decorator to require safe input data"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        input_data = kwargs.get('data', {})
        
        # Check for suspicious content
        for field_name, field_value in input_data.items():
            if isinstance(field_value, str):
                suspicious_patterns = has_suspicious_content(field_value)
                if suspicious_patterns:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Suspicious content detected in {field_name}"
                    )
        
        return await func(*args, **kwargs)
    
    return wrapper

def require_rate_limit_compliance(max_requests: int, window_seconds: int) -> Callable:
    """Decorator to require rate limit compliance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            client_ip = kwargs.get('client_ip', 'unknown')
            request_history = kwargs.get('request_history', {})
            
            if has_exceeded_rate_limit(client_ip, request_history, max_requests, window_seconds):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage with descriptive naming

async def main():
    """Example usage of descriptive security system"""
    print("🔒 Descriptive Security Examples")
    
    # Initialize security manager
    security_manager = DescriptiveSecurityManager("secret-key", "encryption-key")
    
    # Test user registration validation
    registration_data = {
        "email": "user@example.com",
        "password": "WeakPass"
    }
    
    validator = DescriptiveSecurityValidator()
    registration_validation = validator.validate_user_registration_data(registration_data)
    
    print("📊 Registration validation results:")
    for field_name, validation_result in registration_validation.items():
        if "is_valid" in validation_result:
            is_field_valid = validation_result["is_valid"]
            print(f"   {field_name}: {'✅' if is_field_valid else '❌'}")
        elif "is_password_valid" in validation_result:
            is_password_valid = validation_result["is_password_valid"]
            print(f"   {field_name}: {'✅' if is_password_valid else '❌'}")
    
    # Test login security
    login_result = security_manager.is_user_login_allowed(
        "user@example.com", "WeakPass", "127.0.0.1"
    )
    
    print(f"🔐 Login security check:")
    print(f"   Is login allowed: {'✅' if login_result['is_login_allowed'] else '❌'}")
    print(f"   Reason: {login_result.get('reason', 'N/A')}")
    
    # Test video processing security
    video_data = {
        "title": "My Video<script>alert('xss')</script>",
        "url": "https://example.com/video.mp4",
        "description": "A great video with SQL injection: ' OR 1=1 --"
    }
    
    processing_result = security_manager.is_video_processing_safe(
        video_data, "valid-token"
    )
    
    print(f"🎥 Video processing security check:")
    print(f"   Is processing safe: {'✅' if processing_result['is_processing_safe'] else '❌'}")
    print(f"   Reason: {processing_result.get('reason', 'N/A')}")
    
    # Demonstrate descriptive decorators
    @require_valid_authentication
    @require_safe_input_data
    @require_rate_limit_compliance(max_requests=10, window_seconds=60)
    async def secure_endpoint(data: Dict, authenticated_user: Dict, client_ip: str):
        return {
            "success": True,
            "message": "Secure endpoint accessed",
            "user": authenticated_user["sub"]
        }
    
    print("🎯 Descriptive security examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 