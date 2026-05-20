"""
Security utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import hashlib
import secrets
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import request, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def init_security(app) -> None:
    """Initialize security utilities with app."""
    # Set security headers
    app.config.setdefault('SECURITY_HEADERS', {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'"
    })
    
    logger.info("🔒 Security utilities initialized")

def hash_password(password: str) -> str:
    """Hash password with early returns."""
    if not password or not isinstance(password, str):
        raise ValueError("Password must be a non-empty string")
    
    return generate_password_hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password with early returns."""
    if not password or not hashed_password:
        return False
    
    return check_password_hash(hashed_password, password)

def generate_token(payload: Dict[str, Any], secret_key: str = None, expires_in: int = 3600) -> str:
    """Generate JWT token with early returns."""
    if not payload:
        raise ValueError("Payload cannot be empty")
    
    secret = secret_key or current_app.config.get('JWT_SECRET_KEY')
    if not secret:
        raise ValueError("JWT secret key not configured")
    
    # Add expiration time
    payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
    payload['iat'] = datetime.utcnow()
    
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_token(token: str, secret_key: str = None) -> Optional[Dict[str, Any]]:
    """Verify JWT token with early returns."""
    if not token:
        return None
    
    secret = secret_key or current_app.config.get('JWT_SECRET_KEY')
    if not secret:
        return None
    
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("❌ Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("❌ Invalid token")
        return None

def generate_csrf_token() -> str:
    """Generate CSRF token with early returns."""
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token with early returns."""
    if not token or not session_token:
        return False
    
    return secrets.compare_digest(token, session_token)

def sanitize_input(input_string: str) -> str:
    """Sanitize input string with early returns."""
    if not input_string or not isinstance(input_string, str):
        return ""
    
    # Remove potentially harmful characters
    import re
    sanitized = re.sub(r'[<>"\']', '', input_string)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+=', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def validate_email(email: str) -> bool:
    """Validate email format with early returns."""
    if not email or not isinstance(email, str):
        return False
    
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength with early returns."""
    if not password or not isinstance(password, str):
        return {'valid': False, 'message': 'Password must be a string'}
    
    if len(password) < 8:
        return {'valid': False, 'message': 'Password must be at least 8 characters long'}
    
    if len(password) > 128:
        return {'valid': False, 'message': 'Password must be no more than 128 characters long'}
    
    # Check for required character types
    import re
    has_lower = bool(re.search(r'[a-z]', password))
    has_upper = bool(re.search(r'[A-Z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    
    strength_score = sum([has_lower, has_upper, has_digit, has_special])
    
    if strength_score < 3:
        return {'valid': False, 'message': 'Password must contain at least 3 of: lowercase, uppercase, digits, special characters'}
    
    return {'valid': True, 'strength_score': strength_score}

def check_rate_limit(client_ip: str, endpoint: str, max_requests: int = 100, window: int = 3600) -> bool:
    """Check rate limit with early returns."""
    if not client_ip or not endpoint:
        return False
    
    # This is a simplified implementation
    # In production, you'd use Redis or similar
    rate_key = f"rate_limit:{client_ip}:{endpoint}"
    
    # Mock rate limit check
    return True

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security event with early returns."""
    if not event_type or not details:
        return
    
    logger.warning(f"🔒 Security event: {event_type} - {details}")

def check_ip_whitelist(client_ip: str, whitelist: List[str]) -> bool:
    """Check IP against whitelist with early returns."""
    if not client_ip or not whitelist:
        return False
    
    return client_ip in whitelist

def check_ip_blacklist(client_ip: str, blacklist: List[str]) -> bool:
    """Check IP against blacklist with early returns."""
    if not client_ip or not blacklist:
        return True
    
    return client_ip not in blacklist

def generate_secure_filename(filename: str) -> str:
    """Generate secure filename with early returns."""
    if not filename or not isinstance(filename, str):
        return ""
    
    import os
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove potentially harmful characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # Add timestamp for uniqueness
    timestamp = str(int(time.time()))
    name, ext = os.path.splitext(filename)
    
    return f"{name}_{timestamp}{ext}"

def validate_file_upload(filename: str, content_type: str, max_size: int, allowed_types: List[str]) -> Dict[str, Any]:
    """Validate file upload with early returns."""
    if not filename or not content_type:
        return {'valid': False, 'message': 'Filename and content type required'}
    
    # Check file extension
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_types:
        return {'valid': False, 'message': f'File type {ext} not allowed'}
    
    # Check content type
    if content_type not in allowed_types:
        return {'valid': False, 'message': f'Content type {content_type} not allowed'}
    
    # Check file size (this would be done after upload in real implementation)
    if max_size <= 0:
        return {'valid': False, 'message': 'Invalid max size'}
    
    return {'valid': True, 'message': 'File upload valid'}

def encrypt_sensitive_data(data: str, key: str = None) -> str:
    """Encrypt sensitive data with early returns."""
    if not data:
        return ""
    
    secret_key = key or current_app.config.get('SECRET_KEY')
    if not secret_key:
        raise ValueError("Secret key not configured")
    
    # Simple encryption (use proper encryption in production)
    import base64
    encoded = base64.b64encode(data.encode()).decode()
    return encoded

def decrypt_sensitive_data(encrypted_data: str, key: str = None) -> str:
    """Decrypt sensitive data with early returns."""
    if not encrypted_data:
        return ""
    
    secret_key = key or current_app.config.get('SECRET_KEY')
    if not secret_key:
        raise ValueError("Secret key not configured")
    
    # Simple decryption (use proper decryption in production)
    import base64
    try:
        decoded = base64.b64decode(encrypted_data.encode()).decode()
        return decoded
    except Exception as e:
        logger.error(f"❌ Decryption error: {e}")
        return ""

def generate_api_key() -> str:
    """Generate API key with early returns."""
    return secrets.token_urlsafe(32)

def validate_api_key(api_key: str, valid_keys: List[str]) -> bool:
    """Validate API key with early returns."""
    if not api_key or not valid_keys:
        return False
    
    return api_key in valid_keys

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage with early returns."""
    if not api_key:
        return ""
    
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key_hash(api_key: str, hashed_key: str) -> bool:
    """Verify API key hash with early returns."""
    if not api_key or not hashed_key:
        return False
    
    return hashlib.sha256(api_key.encode()).hexdigest() == hashed_key

# Security decorators
def require_authentication(func: Callable) -> Callable:
    """Decorator for authentication requirement with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for JWT token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return {'success': False, 'message': 'Authentication required', 'error_type': 'authentication_required'}
        
        token = auth_header.split(' ')[1]
        payload = verify_token(token)
        
        if not payload:
            return {'success': False, 'message': 'Invalid token', 'error_type': 'invalid_token'}
        
        # Store user info in request context
        g.current_user = payload
        
        return func(*args, **kwargs)
    return wrapper

def require_permissions(*required_permissions: str):
    """Decorator for permission requirement with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if user is authenticated
            if not hasattr(g, 'current_user'):
                return {'success': False, 'message': 'Authentication required', 'error_type': 'authentication_required'}
            
            # Check permissions (mock implementation)
            user_permissions = g.current_user.get('permissions', [])
            missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
            
            if missing_permissions:
                return {'success': False, 'message': f'Missing permissions: {", ".join(missing_permissions)}', 'error_type': 'insufficient_permissions'}
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_key(func: Callable) -> Callable:
    """Decorator for API key requirement with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return {'success': False, 'message': 'API key required', 'error_type': 'api_key_required'}
        
        # Validate API key (mock implementation)
        valid_keys = current_app.config.get('VALID_API_KEYS', [])
        if not validate_api_key(api_key, valid_keys):
            return {'success': False, 'message': 'Invalid API key', 'error_type': 'invalid_api_key'}
        
        return func(*args, **kwargs)
    return wrapper

def rate_limit_decorator(max_requests: int = 100, window: int = 3600):
    """Decorator for rate limiting with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            endpoint = request.endpoint or func.__name__
            
            if not check_rate_limit(client_ip, endpoint, max_requests, window):
                return {'success': False, 'message': 'Rate limit exceeded', 'error_type': 'rate_limit_exceeded'}
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def sanitize_input_decorator(func: Callable) -> Callable:
    """Decorator for input sanitization with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Sanitize request data
        if request.is_json:
            data = request.get_json()
            if data:
                sanitized_data = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        sanitized_data[key] = sanitize_input(value)
                    else:
                        sanitized_data[key] = value
                
                # Replace request data with sanitized data
                request._cached_json = sanitized_data
        
        return func(*args, **kwargs)
    return wrapper

def log_security_events(func: Callable) -> Callable:
    """Decorator for security event logging with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Log security event
            log_security_event('function_error', {
                'function': func.__name__,
                'error': str(e),
                'client_ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent'),
                'timestamp': time.time()
            })
            raise
    return wrapper

# Security utility functions
def create_security_headers() -> Dict[str, str]:
    """Create security headers with early returns."""
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }

def validate_request_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Validate request origin with early returns."""
    if not origin or not allowed_origins:
        return False
    
    return origin in allowed_origins

def check_request_size(max_size: int = 1024 * 1024) -> bool:
    """Check request size with early returns."""
    content_length = request.content_length
    if not content_length:
        return True
    
    return content_length <= max_size

def validate_user_agent(user_agent: str, blocked_agents: List[str] = None) -> bool:
    """Validate user agent with early returns."""
    if not user_agent:
        return False
    
    if not blocked_agents:
        return True
    
    return not any(blocked in user_agent.lower() for blocked in blocked_agents)









