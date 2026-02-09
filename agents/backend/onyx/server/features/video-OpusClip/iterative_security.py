#!/usr/bin/env python3
"""
Iterative Security Implementation for Video-OpusClip
Uses iteration and reusable patterns to avoid code duplication
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
    secret_key: str = "your-secret-key"
    encryption_key: str = "your-encryption-key"
    salt: str = "your-salt"
    max_login_attempts: int = 5
    lockout_duration: int = 900
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

# Security levels
class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Iterative validation patterns
VALIDATION_PATTERNS = {
    "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "password": {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digit": True,
        "require_special": True
    },
    "url": {
        "allowed_schemes": ["http://", "https://"],
        "blocked_patterns": ["javascript:", "data:", "vbscript:", "file:", "ftp:"]
    },
    "input": {
        "max_length": 1000,
        "blocked_chars": ['<', '>', '"', "'"],
        "blocked_patterns": [r'<script.*?</script>']
    }
}

# Iterative validation functions
def validate_field(value: str, field_type: str) -> Dict[str, Any]:
    """Generic field validation using patterns"""
    import re
    
    if field_type == "email":
        pattern = VALIDATION_PATTERNS["email"]
        is_valid = bool(re.match(pattern, value))
        return {"valid": is_valid, "errors": [] if is_valid else ["Invalid email format"]}
    
    elif field_type == "password":
        rules = VALIDATION_PATTERNS["password"]
        errors = []
        
        if len(value) < rules["min_length"]:
            errors.append(f"Password must be at least {rules['min_length']} characters")
        
        if rules["require_uppercase"] and not re.search(r'[A-Z]', value):
            errors.append("Password must contain uppercase letter")
        
        if rules["require_lowercase"] and not re.search(r'[a-z]', value):
            errors.append("Password must contain lowercase letter")
        
        if rules["require_digit"] and not re.search(r'\d', value):
            errors.append("Password must contain digit")
        
        if rules["require_special"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            errors.append("Password must contain special character")
        
        return {"valid": not errors, "errors": errors}
    
    elif field_type == "url":
        rules = VALIDATION_PATTERNS["url"]
        errors = []
        
        # Check allowed schemes
        if not any(value.startswith(scheme) for scheme in rules["allowed_schemes"]):
            errors.append("URL must start with http:// or https://")
        
        # Check blocked patterns
        for pattern in rules["blocked_patterns"]:
            if pattern.lower() in value.lower():
                errors.append(f"URL contains blocked pattern: {pattern}")
        
        return {"valid": not errors, "errors": errors}
    
    elif field_type == "input":
        rules = VALIDATION_PATTERNS["input"]
        errors = []
        
        if len(value) > rules["max_length"]:
            errors.append(f"Input too long (max {rules['max_length']} characters)")
        
        # Check blocked characters
        for char in rules["blocked_chars"]:
            if char in value:
                errors.append(f"Input contains blocked character: {char}")
        
        # Check blocked patterns
        for pattern in rules["blocked_patterns"]:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"Input contains blocked pattern: {pattern}")
        
        return {"valid": not errors, "errors": errors}
    
    return {"valid": True, "errors": []}

def validate_multiple_fields(data: Dict[str, Any], field_validations: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Validate multiple fields using iteration"""
    results = {}
    
    for field, validation_type in field_validations.items():
        value = data.get(field, "")
        results[field] = validate_field(value, validation_type)
    
    return results

def sanitize_input(text: str) -> str:
    """Sanitize input using iterative pattern replacement"""
    import re
    
    # Iterate through blocked characters
    for char in VALIDATION_PATTERNS["input"]["blocked_chars"]:
        text = text.replace(char, '')
    
    # Iterate through blocked patterns
    for pattern in VALIDATION_PATTERNS["input"]["blocked_patterns"]:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

# Iterative encryption/decryption
class CryptoManager:
    """Iterative encryption management"""
    
    def __init__(self, key: str):
        self.cipher = Fernet(key.encode())
    
    def encrypt(self, data: str) -> str:
        """Encrypt single value"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt single value"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_fields(self, data: Dict[str, Any], fields_to_encrypt: List[str]) -> Dict[str, Any]:
        """Iteratively encrypt multiple fields"""
        encrypted_data = data.copy()
        
        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        
        return encrypted_data
    
    def decrypt_fields(self, data: Dict[str, Any], fields_to_decrypt: List[str]) -> Dict[str, Any]:
        """Iteratively decrypt multiple fields"""
        decrypted_data = data.copy()
        
        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
        
        return decrypted_data

# Iterative JWT management
class JWTManager:
    """Iterative JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, data: Dict[str, Any], expires_minutes: int = 30) -> str:
        """Create single JWT token"""
        payload = data.copy()
        payload.update({
            "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
            "iat": datetime.utcnow()
        })
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_multiple_tokens(self, data: Dict[str, Any], token_configs: Dict[str, int]) -> Dict[str, str]:
        """Iteratively create multiple tokens"""
        tokens = {}
        
        for token_type, expires_minutes in token_configs.items():
            token_data = data.copy()
            if token_type == "refresh":
                token_data["type"] = "refresh"
            
            tokens[f"{token_type}_token"] = self.create_token(token_data, expires_minutes)
        
        return tokens
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify single JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Iterative rate limiting
class RateLimiter:
    """Iterative rate limiting system"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Iteratively remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        self.requests[identifier].append(now)
        return True
    
    def get_multiple_limits(self, identifiers: List[str]) -> Dict[str, bool]:
        """Check rate limits for multiple identifiers"""
        results = {}
        
        for identifier in identifiers:
            results[identifier] = self.is_allowed(identifier)
        
        return results

# Iterative intrusion detection
class IntrusionDetector:
    """Iterative intrusion detection system"""
    
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
    
    def check_multiple_ips(self, ip_attempts: List[tuple[str, bool]]) -> Dict[str, bool]:
        """Check multiple IP login attempts"""
        results = {}
        
        for ip, success in ip_attempts:
            results[ip] = self.check_login_attempt(ip, success)
        
        return results
    
    def check_login_attempt(self, ip: str, success: bool) -> bool:
        """Check single login attempt"""
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
        """Check if single IP is blocked"""
        if ip not in self.blocked_ips:
            return False
        
        if time.time() - self.blocked_ips[ip] > self.lockout_duration:
            del self.blocked_ips[ip]
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            return False
        
        return True
    
    def check_multiple_ips_blocked(self, ips: List[str]) -> Dict[str, bool]:
        """Check if multiple IPs are blocked"""
        results = {}
        
        for ip in ips:
            results[ip] = self.is_ip_blocked(ip)
        
        return results
    
    def detect_suspicious_activity(self, data: str) -> List[str]:
        """Iteratively detect suspicious patterns"""
        import re
        detected = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                detected.append(pattern)
        
        return detected

# Iterative logging system
class SecurityLogger:
    """Iterative security logging system"""
    
    def __init__(self, log_file: str = "security.log"):
        self.log_file = log_file
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log single security event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        print(f"SECURITY_LOG: {log_entry}")
    
    def log_multiple_events(self, events: List[tuple[str, Dict[str, Any]]]) -> None:
        """Iteratively log multiple events"""
        for event_type, details in events:
            self.log_event(event_type, details)
    
    def log_access_attempts(self, attempts: List[tuple[str, str, str, bool, str]]) -> None:
        """Iteratively log multiple access attempts"""
        for user, resource, action, success, ip in attempts:
            self.log_event("ACCESS", {
                "user": user,
                "resource": resource,
                "action": action,
                "success": success,
                "ip": ip
            })

# Iterative decorator system
class SecurityDecorators:
    """Iterative security decorators"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.jwt_manager = JWTManager(config.secret_key)
        self.rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        self.intrusion_detector = IntrusionDetector(config.max_login_attempts, config.lockout_duration)
        self.logger = SecurityLogger()
    
    def apply_multiple_decorators(self, func: Callable, decorators: List[Callable]) -> Callable:
        """Apply multiple decorators iteratively"""
        decorated_func = func
        
        for decorator in decorators:
            decorated_func = decorator(decorated_func)
        
        return decorated_func
    
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
    
    def validate_input(self, field_validations: Dict[str, str]) -> Callable:
        """Input validation decorator"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if 'data' in kwargs:
                    validation_results = validate_multiple_fields(kwargs['data'], field_validations)
                    kwargs['validation_results'] = validation_results
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Iterative user management
class UserManager:
    """Iterative user management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.crypto = CryptoManager(config.encryption_key)
        self.jwt_manager = JWTManager(config.secret_key)
        self.logger = SecurityLogger()
        self.users = {}
    
    def register_multiple_users(self, user_data: List[tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Iteratively register multiple users"""
        results = []
        
        for email, password, client_ip in user_data:
            try:
                result = self.register_user(email, password, client_ip)
                results.append({"success": True, "data": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return results
    
    def register_user(self, email: str, password: str, client_ip: str) -> Dict[str, Any]:
        """Register single user"""
        # Validate input using iterative validation
        field_validations = {
            "email": "email",
            "password": "password"
        }
        
        validation_results = validate_multiple_fields(
            {"email": email, "password": password}, 
            field_validations
        )
        
        # Check validation results
        for field, result in validation_results.items():
            if not result["valid"]:
                raise ValueError(f"Validation failed for {field}: {result['errors']}")
        
        if email in self.users:
            raise ValueError("User already exists")
        
        # Create user
        import hashlib
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), self.config.salt.encode(), 100000).hex()
        
        self.users[email] = {
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "permissions": ["user"]
        }
        
        self.logger.log_event("ACCESS", {
            "user": email,
            "resource": "/auth/register",
            "action": "register",
            "success": True,
            "ip": client_ip
        })
        
        return {
            "success": True,
            "message": "User registered successfully",
            "data": {"email": email}
        }
    
    def authenticate_multiple_users(self, auth_data: List[tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Iteratively authenticate multiple users"""
        results = []
        
        for email, password, client_ip in auth_data:
            try:
                tokens = self.authenticate_user(email, password, client_ip)
                results.append({"success": True, "tokens": tokens})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return results
    
    def authenticate_user(self, email: str, password: str, client_ip: str) -> Optional[Dict[str, str]]:
        """Authenticate single user"""
        if email not in self.users:
            self.logger.log_event("ACCESS", {
                "user": "unknown",
                "resource": "/auth/login",
                "action": "login",
                "success": False,
                "ip": client_ip
            })
            return None
        
        import hashlib
        user = self.users[email]
        hashed_input = hashlib.pbkdf2_hmac('sha256', password.encode(), self.config.salt.encode(), 100000).hex()
        
        if hashed_input != user["hashed_password"]:
            self.logger.log_event("ACCESS", {
                "user": email,
                "resource": "/auth/login",
                "action": "login",
                "success": False,
                "ip": client_ip
            })
            return None
        
        # Success
        self.logger.log_event("ACCESS", {
            "user": email,
            "resource": "/auth/login",
            "action": "login",
            "success": True,
            "ip": client_ip
        })
        
        # Generate tokens using iterative token creation
        token_configs = {
            "access": 30,
            "refresh": 1440
        }
        
        tokens = self.jwt_manager.create_multiple_tokens({
            "sub": email,
            "permissions": user["permissions"]
        }, token_configs)
        
        return tokens

# Example usage with iterative patterns
async def main():
    """Example usage of iterative security system"""
    print("🔒 Iterative Security System Example")
    
    # Initialize configuration
    config = SecurityConfig()
    
    # Initialize components
    user_manager = UserManager(config)
    decorators = SecurityDecorators(config)
    
    # Register multiple users iteratively
    user_data = [
        ("user1@example.com", "SecurePass123!", "127.0.0.1"),
        ("user2@example.com", "SecurePass456!", "127.0.0.2"),
        ("user3@example.com", "SecurePass789!", "127.0.0.3")
    ]
    
    registration_results = user_manager.register_multiple_users(user_data)
    print(f"✅ Registered {len([r for r in registration_results if r['success']])} users")
    
    # Authenticate multiple users iteratively
    auth_data = [
        ("user1@example.com", "SecurePass123!", "127.0.0.1"),
        ("user2@example.com", "SecurePass456!", "127.0.0.2"),
        ("user3@example.com", "SecurePass789!", "127.0.0.3")
    ]
    
    auth_results = user_manager.authenticate_multiple_users(auth_data)
    print(f"✅ Authenticated {len([r for r in auth_results if r['success']])} users")
    
    # Demonstrate iterative validation
    test_data = {
        "title": "My Video<script>alert('xss')</script>",
        "url": "https://example.com/video.mp4",
        "description": "A great video with <script> tags"
    }
    
    field_validations = {
        "title": "input",
        "url": "url",
        "description": "input"
    }
    
    validation_results = validate_multiple_fields(test_data, field_validations)
    print("📊 Validation results:")
    for field, result in validation_results.items():
        print(f"   {field}: {'✅' if result['valid'] else '❌'} {result['errors']}")
    
    # Demonstrate iterative decorators
    def sample_endpoint(data: Dict, current_user: Dict, validation_results: Dict):
        return {"success": True, "data": data}
    
    # Apply multiple decorators iteratively
    secure_endpoint = decorators.apply_multiple_decorators(
        sample_endpoint,
        [
            decorators.require_auth,
            decorators.rate_limit(max_requests=10, window=60),
            decorators.validate_input(field_validations)
        ]
    )
    
    print("🎯 Iterative security system ready!")

if __name__ == "__main__":
    asyncio.run(main()) 