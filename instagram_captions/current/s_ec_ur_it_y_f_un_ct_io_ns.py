from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import hashlib
import hmac
import time
import re
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Tuple
from functools import wraps, partial
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Request, Depends, status
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Functional Security Implementation for Instagram Captions API

Pure functions, decorators, and declarative patterns for security.
No classes - functional programming approach.
"""


# Configuration
SECRET_KEY = "your-secret-key"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Structured logging
logger = structlog.get_logger()


# =============================================================================
# PURE FUNCTIONS - AUTHENTICATION
# =============================================================================

def hash_password(plain_password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def generate_api_key(user_id: str) -> str:
    """Generate secure API key."""
    timestamp = str(int(time.time()))
    message = f"{user_id}:{timestamp}"
    signature = hmac.new(
        SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{user_id}.{timestamp}.{signature}"


async def validate_api_key(api_key: str) -> bool:
    """Validate API key format and signature."""
    try:
        parts = api_key.split('.')
        if len(parts) != 3:
            return False
        
        user_id, timestamp, signature = parts
        message = f"{user_id}:{timestamp}"
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Check if key is not expired (24 hours)
        if int(time.time()) - int(timestamp) > 86400:
            return False
        
        return hmac.compare_digest(signature, expected_signature)
    except (ValueError, IndexError):
        return False


# =============================================================================
# PURE FUNCTIONS - RATE LIMITING
# =============================================================================

def check_rate_limit(user_id: str, requests_per_minute: int = RATE_LIMIT_REQUESTS) -> bool:
    """Check if user has exceeded rate limit."""
    key = f"rate_limit:{user_id}"
    current = redis_client.get(key)
    
    if current and int(current) >= requests_per_minute:
        return False
    
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, RATE_LIMIT_WINDOW)
    pipe.execute()
    return True


def enforce_rate_limit(user_id: str) -> None:
    """Enforce rate limiting - raise exception if exceeded."""
    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
        )


def get_rate_limit_info(user_id: str) -> Dict[str, Any]:
    """Get current rate limit information."""
    key = f"rate_limit:{user_id}"
    current = redis_client.get(key)
    ttl = redis_client.ttl(key)
    
    return {
        "current_requests": int(current) if current else 0,
        "limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "remaining_ttl": ttl if ttl > 0 else 0
    }


# =============================================================================
# PURE FUNCTIONS - INPUT VALIDATION
# =============================================================================

def sanitize_content(content: str) -> str:
    """Sanitize user content to prevent XSS."""
    # Remove script tags
    content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
    # Remove javascript: protocol
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
    # Remove other dangerous protocols
    content = re.sub(r'(data|vbscript|file):', '', content, flags=re.IGNORECASE)
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Trim whitespace
    return content.strip()


def validate_content_description(description: str) -> Tuple[bool, str]:
    """Validate content description."""
    if not description or len(description.strip()) < 10:
        return False, "Content description must be at least 10 characters"
    
    if len(description) > 1000:
        return False, "Content description must be less than 1000 characters"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'file:'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return False, f"Content contains forbidden pattern: {pattern}"
    
    return True, "Valid"


def validate_style(style: str) -> Tuple[bool, str]:
    """Validate caption style."""
    allowed_styles = ['casual', 'formal', 'creative', 'professional', 'funny', 'inspirational']
    if style not in allowed_styles:
        return False, f"Style must be one of: {', '.join(allowed_styles)}"
    return True, "Valid"


def validate_hashtag_count(count: int) -> Tuple[bool, str]:
    """Validate hashtag count."""
    if not isinstance(count, int):
        return False, "Hashtag count must be an integer"
    if count < 0 or count > 30:
        return False, "Hashtag count must be between 0 and 30"
    return True, "Valid"


async def validate_request_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """Validate complete request data."""
    errors = []
    
    # Validate content description
    if 'content_description' not in data:
        errors.append("content_description is required")
    else:
        is_valid, message = validate_content_description(data['content_description'])
        if not is_valid:
            errors.append(f"content_description: {message}")
        else:
            data['content_description'] = sanitize_content(data['content_description'])
    
    # Validate style
    if 'style' in data:
        is_valid, message = validate_style(data['style'])
        if not is_valid:
            errors.append(f"style: {message}")
    
    # Validate hashtag count
    if 'hashtag_count' in data:
        is_valid, message = validate_hashtag_count(data['hashtag_count'])
        if not is_valid:
            errors.append(f"hashtag_count: {message}")
    
    if errors:
        return False, data, "; ".join(errors)
    
    return True, data, "Valid"


# =============================================================================
# PURE FUNCTIONS - SECURITY LOGGING
# =============================================================================

def log_security_event(event_type: str, user_id: str, details: Dict[str, Any] = None) -> None:
    """Log security event."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details or {},
        "severity": "info"
    }
    logger.info("security_event", **log_entry)


def log_authentication_attempt(user_id: str, success: bool, ip_address: str = None) -> None:
    """Log authentication attempt."""
    log_security_event(
        "authentication_attempt",
        user_id,
        {
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_rate_limit_violation(user_id: str, endpoint: str) -> None:
    """Log rate limit violation."""
    log_security_event(
        "rate_limit_violation",
        user_id,
        {
            "endpoint": endpoint,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_input_validation_error(user_id: str, error_message: str) -> None:
    """Log input validation error."""
    log_security_event(
        "input_validation_error",
        user_id,
        {
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# FUNCTIONAL DECORATORS
# =============================================================================

def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Extract token from kwargs or request
        token = kwargs.get('token')
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Verify token
        user_data = verify_token(token)
        kwargs['current_user'] = user_data
        
        return await func(*args, **kwargs)
    return wrapper


def require_rate_limit(func: Callable) -> Callable:
    """Decorator to enforce rate limiting."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        user_id = kwargs.get('current_user', {}).get('user_id') or kwargs.get('api_key')
        if user_id:
            enforce_rate_limit(user_id)
        return await func(*args, **kwargs)
    return wrapper


def validate_input(func: Callable) -> Callable:
    """Decorator to validate input data."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        request_data = kwargs.get('request_data') or kwargs.get('data')
        if request_data:
            is_valid, cleaned_data, error_message = validate_request_data(request_data)
            if not is_valid:
                log_input_validation_error(
                    kwargs.get('current_user', {}).get('user_id', 'unknown'),
                    error_message
                )
                raise HTTPException(status_code=400, detail=error_message)
            kwargs['request_data'] = cleaned_data
        
        return await func(*args, **kwargs)
    return wrapper


def log_security_events(event_type: str) -> Callable:
    """Decorator to log security events."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            # Log the event
            user_id = kwargs.get('current_user', {}).get('user_id') or kwargs.get('api_key', 'unknown')
            log_security_event(event_type, user_id, {
                "function": func.__name__,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
        return wrapper
    return decorator


# =============================================================================
# FUNCTIONAL DEPENDENCIES FOR FASTAPI
# =============================================================================

def get_current_user(token: str) -> Dict[str, Any]:
    """FastAPI dependency to get current user from token."""
    return verify_token(token)


def get_rate_limit_info_dependency(user_id: str) -> Dict[str, Any]:
    """FastAPI dependency to get rate limit information."""
    return get_rate_limit_info(user_id)


def enforce_rate_limit_dependency(user_id: str) -> None:
    """FastAPI dependency to enforce rate limiting."""
    return enforce_rate_limit(user_id)


# =============================================================================
# PURE FUNCTIONS - SECURITY HEADERS
# =============================================================================

def add_security_headers(response_headers: Dict[str, str]) -> Dict[str, str]:
    """Add security headers to response."""
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
    }
    
    response_headers.update(security_headers)
    return response_headers


def add_cache_headers(response_headers: Dict[str, str], cache_control: str = "no-cache") -> Dict[str, str]:
    """Add cache control headers."""
    cache_headers = {
        "Cache-Control": cache_control,
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    response_headers.update(cache_headers)
    return response_headers


# =============================================================================
# PURE FUNCTIONS - UTILITY
# =============================================================================

async def generate_request_id() -> str:
    """Generate unique request ID."""
    return f"req_{int(time.time() * 1000)}_{hash(str(time.time()))}"


def mask_sensitive_data(data: str, mask_char: str = "*") -> str:
    """Mask sensitive data for logging."""
    if len(data) <= 4:
        return mask_char * len(data)
    return data[:2] + mask_char * (len(data) - 4) + data[-2:]


async def is_suspicious_request(request_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if request is suspicious."""
    suspicious_patterns = [
        (r'<script', "Script tag detected"),
        (r'javascript:', "JavaScript protocol detected"),
        (r'data:text/html', "Data URI detected"),
        (r'vbscript:', "VBScript protocol detected"),
        (r'file:', "File protocol detected"),
        (r'../../', "Path traversal attempt"),
        (r'%00', "Null byte injection"),
        (r'<iframe', "Iframe tag detected")
    ]
    
    request_str = json.dumps(request_data, default=str)
    
    for pattern, description in suspicious_patterns:
        if re.search(pattern, request_str, re.IGNORECASE):
            return True, description
    
    return False, ""


async def calculate_request_hash(request_data: Dict[str, Any]) -> str:
    """Calculate hash of request data for deduplication."""
    data_str = json.dumps(request_data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# FUNCTIONAL MIDDLEWARE FACTORY
# =============================================================================

def create_security_middleware() -> Callable:
    """Create security middleware function."""
    async def security_middleware(request: Request, call_next):
        
    """security_middleware function."""
# Add request ID
        request.state.request_id = generate_request_id()
        request.state.start_time = time.time()
        
        # Check for suspicious requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_data = json.loads(body)
                    is_suspicious, reason = is_suspicious_request(request_data)
                    if is_suspicious:
                        log_security_event(
                            "suspicious_request",
                            "unknown",
                            {"reason": reason, "request_id": request.state.request_id}
                        )
                        raise HTTPException(status_code=400, detail="Invalid request")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers.update(add_security_headers(dict(response.headers)))
        
        # Add processing time
        processing_time = time.time() - request.state.start_time
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response
    
    return security_middleware


# =============================================================================
# EXPORT ALL FUNCTIONS
# =============================================================================

__all__ = [
    # Authentication
    "hash_password", "verify_password", "create_access_token", "verify_token",
    "generate_api_key", "validate_api_key",
    
    # Rate limiting
    "check_rate_limit", "enforce_rate_limit", "get_rate_limit_info",
    
    # Input validation
    "sanitize_content", "validate_content_description", "validate_style",
    "validate_hashtag_count", "validate_request_data",
    
    # Security logging
    "log_security_event", "log_authentication_attempt", "log_rate_limit_violation",
    "log_input_validation_error",
    
    # Decorators
    "require_authentication", "require_rate_limit", "validate_input", "log_security_events",
    
    # Dependencies
    "get_current_user", "get_rate_limit_info_dependency", "enforce_rate_limit_dependency",
    
    # Headers
    "add_security_headers", "add_cache_headers",
    
    # Utilities
    "generate_request_id", "mask_sensitive_data", "is_suspicious_request",
    "calculate_request_hash",
    
    # Middleware
    "create_security_middleware"
] 