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
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from functools import wraps, partial, reduce
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Request, Depends, status
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Functional Core Module - Instagram Captions API

Modular functional implementation with descriptive variable names using auxiliary verbs.
Eliminates code duplication through reusable functional modules.
"""


# Configuration constants
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
# FUNCTIONAL UTILITIES - REUSABLE MODULES
# =============================================================================

def create_pipe(*functions: Callable) -> Callable:
    """Create functional pipe for function composition."""
    def pipe_inner(value) -> Any:
        return reduce(lambda accumulated_value, current_function: current_function(accumulated_value), functions, value)
    return pipe_inner


def create_curry(func: Callable, *args, **kwargs) -> Callable:
    """Create curried function with partial arguments."""
    return partial(func, *args, **kwargs)


def create_map_function(transformation_func: Callable) -> Callable:
    """Create map function for list transformations."""
    return lambda items: list(map(transformation_func, items))


def create_filter_function(condition_func: Callable) -> Callable:
    """Create filter function for list filtering."""
    return lambda items: list(filter(condition_func, items))


def create_reduce_function(reduction_func: Callable, initial_value=None) -> Callable:
    """Create reduce function for list aggregation."""
    return lambda items: reduce(reduction_func, items, initial_value)


# =============================================================================
# AUTHENTICATION MODULE - DESCRIPTIVE NAMES
# =============================================================================

def hash_password_with_bcrypt(plain_password: str) -> str:
    """Hash password using bcrypt with descriptive naming."""
    return pwd_context.hash(plain_password)


def verify_password_against_hash(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash with descriptive naming."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token_with_expiry(token_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token with expiry."""
    token_payload = token_data.copy()
    expiration_time = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    token_payload.update({"exp": expiration_time})
    return jwt.encode(token_payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token_and_decode(token_string: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        decoded_payload = jwt.decode(token_string, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded_payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def generate_api_key_with_signature(user_identifier: str) -> str:
    """Generate secure API key with HMAC signature."""
    current_timestamp = str(int(time.time()))
    message_to_sign = f"{user_identifier}:{current_timestamp}"
    signature_hash = hmac.new(
        SECRET_KEY.encode(),
        message_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{user_identifier}.{current_timestamp}.{signature_hash}"


async def validate_api_key_signature(api_key_string: str) -> bool:
    """Validate API key format and HMAC signature."""
    try:
        key_components = api_key_string.split('.')
        if len(key_components) != 3:
            return False
        
        user_identifier, timestamp_string, provided_signature = key_components
        message_to_verify = f"{user_identifier}:{timestamp_string}"
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            message_to_verify.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Check if key has expired (24 hours)
        current_time = int(time.time())
        key_timestamp = int(timestamp_string)
        is_key_expired = current_time - key_timestamp > 86400
        
        if is_key_expired:
            return False
        
        return hmac.compare_digest(provided_signature, expected_signature)
    except (ValueError, IndexError):
        return False


# =============================================================================
# RATE LIMITING MODULE - DESCRIPTIVE NAMES
# =============================================================================

def check_rate_limit_for_user(user_identifier: str, requests_per_minute: int = RATE_LIMIT_REQUESTS) -> bool:
    """Check if user has exceeded rate limit."""
    rate_limit_key = f"rate_limit:{user_identifier}"
    current_request_count = redis_client.get(rate_limit_key)
    
    has_exceeded_limit = current_request_count and int(current_request_count) >= requests_per_minute
    
    if has_exceeded_limit:
        return False
    
    # Increment counter and set expiry
    redis_pipeline = redis_client.pipeline()
    redis_pipeline.incr(rate_limit_key)
    redis_pipeline.expire(rate_limit_key, RATE_LIMIT_WINDOW)
    redis_pipeline.execute()
    return True


def enforce_rate_limit_for_user(user_identifier: str) -> None:
    """Enforce rate limiting - raise exception if exceeded."""
    is_within_limit = check_rate_limit_for_user(user_identifier)
    if not is_within_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
        )


def get_rate_limit_information(user_identifier: str) -> Dict[str, Any]:
    """Get current rate limit information."""
    rate_limit_key = f"rate_limit:{user_identifier}"
    current_request_count = redis_client.get(rate_limit_key)
    time_to_live = redis_client.ttl(rate_limit_key)
    
    return {
        "current_requests": int(current_request_count) if current_request_count else 0,
        "limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "remaining_ttl": time_to_live if time_to_live > 0 else 0
    }


# =============================================================================
# INPUT VALIDATION MODULE - DESCRIPTIVE NAMES
# =============================================================================

def sanitize_content_for_xss(content_string: str) -> str:
    """Sanitize user content to prevent XSS attacks."""
    # Remove script tags
    sanitized_content = re.sub(r'<script.*?</script>', '', content_string, flags=re.IGNORECASE | re.DOTALL)
    # Remove javascript protocol
    sanitized_content = re.sub(r'javascript:', '', sanitized_content, flags=re.IGNORECASE)
    # Remove other dangerous protocols
    sanitized_content = re.sub(r'(data|vbscript|file):', '', sanitized_content, flags=re.IGNORECASE)
    # Remove HTML tags
    sanitized_content = re.sub(r'<[^>]+>', '', sanitized_content)
    # Trim whitespace
    return sanitized_content.strip()


def validate_content_description_length(description_text: str) -> Tuple[bool, str]:
    """Validate content description length and format."""
    if not description_text or len(description_text.strip()) < 10:
        return False, "Content description must be at least 10 characters"
    
    if len(description_text) > 1000:
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
        if re.search(pattern, description_text, re.IGNORECASE):
            return False, f"Content contains forbidden pattern: {pattern}"
    
    return True, "Valid"


def validate_caption_style(style_string: str) -> Tuple[bool, str]:
    """Validate caption style against allowed values."""
    allowed_styles = ['casual', 'formal', 'creative', 'professional', 'funny', 'inspirational']
    is_valid_style = style_string in allowed_styles
    
    if not is_valid_style:
        return False, f"Style must be one of: {', '.join(allowed_styles)}"
    return True, "Valid"


def validate_hashtag_count_range(hashtag_count: int) -> Tuple[bool, str]:
    """Validate hashtag count is within acceptable range."""
    if not isinstance(hashtag_count, int):
        return False, "Hashtag count must be an integer"
    
    is_within_range = 0 <= hashtag_count <= 30
    if not is_within_range:
        return False, "Hashtag count must be between 0 and 30"
    
    return True, "Valid"


async def validate_complete_request_data(request_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """Validate complete request data with descriptive error messages."""
    validation_errors = []
    
    # Validate content description
    if 'content_description' not in request_data:
        validation_errors.append("content_description is required")
    else:
        is_content_valid, content_error_message = validate_content_description_length(request_data['content_description'])
        if not is_content_valid:
            validation_errors.append(f"content_description: {content_error_message}")
        else:
            request_data['content_description'] = sanitize_content_for_xss(request_data['content_description'])
    
    # Validate style
    if 'style' in request_data:
        is_style_valid, style_error_message = validate_caption_style(request_data['style'])
        if not is_style_valid:
            validation_errors.append(f"style: {style_error_message}")
    
    # Validate hashtag count
    if 'hashtag_count' in request_data:
        is_hashtag_count_valid, hashtag_error_message = validate_hashtag_count_range(request_data['hashtag_count'])
        if not is_hashtag_count_valid:
            validation_errors.append(f"hashtag_count: {hashtag_error_message}")
    
    has_validation_errors = len(validation_errors) > 0
    if has_validation_errors:
        return False, request_data, "; ".join(validation_errors)
    
    return True, request_data, "Valid"


# =============================================================================
# SECURITY LOGGING MODULE - DESCRIPTIVE NAMES
# =============================================================================

def log_security_event_with_details(event_type: str, user_identifier: str, event_details: Dict[str, Any] = None) -> None:
    """Log security event with comprehensive details."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_identifier,
        "details": event_details or {},
        "severity": "info"
    }
    logger.info("security_event", **log_entry)


def log_authentication_attempt_result(user_identifier: str, is_successful: bool, ip_address: str = None) -> None:
    """Log authentication attempt with result."""
    log_security_event_with_details(
        "authentication_attempt",
        user_identifier,
        {
            "success": is_successful,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_rate_limit_violation_event(user_identifier: str, endpoint_path: str) -> None:
    """Log rate limit violation event."""
    log_security_event_with_details(
        "rate_limit_violation",
        user_identifier,
        {
            "endpoint": endpoint_path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_input_validation_error_event(user_identifier: str, error_message: str) -> None:
    """Log input validation error event."""
    log_security_event_with_details(
        "input_validation_error",
        user_identifier,
        {
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# FUNCTIONAL DECORATORS - DESCRIPTIVE NAMES
# =============================================================================

def require_authentication_token(func: Callable) -> Callable:
    """Decorator to require authentication token."""
    @wraps(func)
    async def authentication_wrapper(*args, **kwargs) -> Any:
        # Extract token from kwargs or request
        authentication_token = kwargs.get('token')
        if not authentication_token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Verify token
        user_data = verify_token_and_decode(authentication_token)
        kwargs['current_user'] = user_data
        
        return await func(*args, **kwargs)
    return authentication_wrapper


def enforce_rate_limiting(func: Callable) -> Callable:
    """Decorator to enforce rate limiting."""
    @wraps(func)
    async def rate_limit_wrapper(*args, **kwargs) -> Any:
        user_identifier = kwargs.get('current_user', {}).get('user_id') or kwargs.get('api_key')
        if user_identifier:
            enforce_rate_limit_for_user(user_identifier)
        return await func(*args, **kwargs)
    return rate_limit_wrapper


def validate_input_data(func: Callable) -> Callable:
    """Decorator to validate input data."""
    @wraps(func)
    async def validation_wrapper(*args, **kwargs) -> Any:
        request_data = kwargs.get('request_data') or kwargs.get('data')
        if request_data:
            is_data_valid, cleaned_data, error_message = validate_complete_request_data(request_data)
            if not is_data_valid:
                log_input_validation_error_event(
                    kwargs.get('current_user', {}).get('user_id', 'unknown'),
                    error_message
                )
                raise HTTPException(status_code=400, detail=error_message)
            kwargs['request_data'] = cleaned_data
        
        return await func(*args, **kwargs)
    return validation_wrapper


def log_security_events_by_type(event_type: str) -> Callable:
    """Decorator to log security events by type."""
    def security_logging_decorator(func: Callable) -> Callable:
        @wraps(func)
        async def logging_wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            # Log the event
            user_identifier = kwargs.get('current_user', {}).get('user_id') or kwargs.get('api_key', 'unknown')
            log_security_event_with_details(event_type, user_identifier, {
                "function": func.__name__,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
        return logging_wrapper
    return security_logging_decorator


# =============================================================================
# FASTAPI DEPENDENCIES - DESCRIPTIVE NAMES
# =============================================================================

def get_current_user_from_token(token_string: str) -> Dict[str, Any]:
    """FastAPI dependency to get current user from token."""
    return verify_token_and_decode(token_string)


def get_rate_limit_information_for_user(user_identifier: str) -> Dict[str, Any]:
    """FastAPI dependency to get rate limit information."""
    return get_rate_limit_information(user_identifier)


def enforce_rate_limiting_for_user(user_identifier: str) -> None:
    """FastAPI dependency to enforce rate limiting."""
    return enforce_rate_limit_for_user(user_identifier)


# =============================================================================
# SECURITY HEADERS MODULE - DESCRIPTIVE NAMES
# =============================================================================

def add_security_headers_to_response(response_headers: Dict[str, str]) -> Dict[str, str]:
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


def add_cache_control_headers_to_response(response_headers: Dict[str, str], cache_control_string: str = "no-cache") -> Dict[str, str]:
    """Add cache control headers to response."""
    cache_headers = {
        "Cache-Control": cache_control_string,
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    response_headers.update(cache_headers)
    return response_headers


# =============================================================================
# UTILITY FUNCTIONS - DESCRIPTIVE NAMES
# =============================================================================

async def generate_unique_request_identifier() -> str:
    """Generate unique request identifier."""
    return f"req_{int(time.time() * 1000)}_{hash(str(time.time()))}"


def mask_sensitive_data_for_logging(data_string: str, mask_character: str = "*") -> str:
    """Mask sensitive data for logging."""
    if len(data_string) <= 4:
        return mask_character * len(data_string)
    return data_string[:2] + mask_character * (len(data_string) - 4) + data_string[-2:]


async def detect_suspicious_request_patterns(request_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Detect suspicious request patterns."""
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
    
    request_string = json.dumps(request_data, default=str)
    
    for pattern, description in suspicious_patterns:
        if re.search(pattern, request_string, re.IGNORECASE):
            return True, description
    
    return False, ""


async def calculate_request_hash_for_deduplication(request_data: Dict[str, Any]) -> str:
    """Calculate hash of request data for deduplication."""
    data_string = json.dumps(request_data, sort_keys=True, default=str)
    return hashlib.sha256(data_string.encode()).hexdigest()


# =============================================================================
# MIDDLEWARE FACTORY - DESCRIPTIVE NAMES
# =============================================================================

def create_security_middleware_function() -> Callable:
    """Create security middleware function."""
    async def security_middleware_handler(request: Request, call_next):
        
    """security_middleware_handler function."""
# Add request identifier
        request.state.request_id = generate_unique_request_identifier()
        request.state.start_time = time.time()
        
        # Check for suspicious requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                request_body = await request.body()
                if request_body:
                    request_data = json.loads(request_body)
                    is_suspicious, suspicious_reason = detect_suspicious_request_patterns(request_data)
                    if is_suspicious:
                        log_security_event_with_details(
                            "suspicious_request",
                            "unknown",
                            {"reason": suspicious_reason, "request_id": request.state.request_id}
                        )
                        raise HTTPException(status_code=400, detail="Invalid request")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers.update(add_security_headers_to_response(dict(response.headers)))
        
        # Add processing time
        processing_time = time.time() - request.state.start_time
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response
    
    return security_middleware_handler


# =============================================================================
# EXPORT ALL FUNCTIONS
# =============================================================================

__all__ = [
    # Functional utilities
    "create_pipe", "create_curry", "create_map_function", "create_filter_function", "create_reduce_function",
    
    # Authentication
    "hash_password_with_bcrypt", "verify_password_against_hash", "create_access_token_with_expiry", 
    "verify_token_and_decode", "generate_api_key_with_signature", "validate_api_key_signature",
    
    # Rate limiting
    "check_rate_limit_for_user", "enforce_rate_limit_for_user", "get_rate_limit_information",
    
    # Input validation
    "sanitize_content_for_xss", "validate_content_description_length", "validate_caption_style",
    "validate_hashtag_count_range", "validate_complete_request_data",
    
    # Security logging
    "log_security_event_with_details", "log_authentication_attempt_result", "log_rate_limit_violation_event",
    "log_input_validation_error_event",
    
    # Decorators
    "require_authentication_token", "enforce_rate_limiting", "validate_input_data", "log_security_events_by_type",
    
    # Dependencies
    "get_current_user_from_token", "get_rate_limit_information_for_user", "enforce_rate_limiting_for_user",
    
    # Headers
    "add_security_headers_to_response", "add_cache_control_headers_to_response",
    
    # Utilities
    "generate_unique_request_identifier", "mask_sensitive_data_for_logging", 
    "detect_suspicious_request_patterns", "calculate_request_hash_for_deduplication",
    
    # Middleware
    "create_security_middleware_function"
] 