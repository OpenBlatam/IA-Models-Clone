from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
import re
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import asyncio
from pydantic import (
from pydantic_core import core_schema, PydanticCustomError
from pydantic.json_schema import JsonSchemaValue
import structlog
from .core.error_handling import error_factory, ValidationError as HeyGenValidationError
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Pydantic v2 Validators for HeyGen AI API
Advanced validation logic with custom validators, performance optimizations, and comprehensive error handling.
"""


    field_validator, model_validator, ValidationError, ValidationInfo,
    PlainValidator, BeforeValidator, AfterValidator, WithJsonSchema,
    GetJsonSchemaHandler, GetCoreSchemaHandler
)


logger = structlog.get_logger()

T = TypeVar('T')

# =============================================================================
# Custom Validation Errors
# =============================================================================

class ValidationErrorCode:
    """Validation error codes for consistent error handling."""
    
    # Content validation
    INVALID_SCRIPT_CONTENT = "INVALID_SCRIPT_CONTENT"
    SCRIPT_TOO_SHORT = "SCRIPT_TOO_SHORT"
    SCRIPT_TOO_LONG = "SCRIPT_TOO_LONG"
    INAPPROPRIATE_CONTENT = "INAPPROPRIATE_CONTENT"
    
    # Format validation
    INVALID_EMAIL_FORMAT = "INVALID_EMAIL_FORMAT"
    INVALID_URL_FORMAT = "INVALID_URL_FORMAT"
    INVALID_PHONE_FORMAT = "INVALID_PHONE_FORMAT"
    INVALID_ID_FORMAT = "INVALID_ID_FORMAT"
    
    # Range validation
    VALUE_TOO_SMALL = "VALUE_TOO_SMALL"
    VALUE_TOO_LARGE = "VALUE_TOO_LARGE"
    DURATION_OUT_OF_RANGE = "DURATION_OUT_OF_RANGE"
    FILE_SIZE_TOO_LARGE = "FILE_SIZE_TOO_LARGE"
    
    # Business logic validation
    INVALID_VOICE_ID = "INVALID_VOICE_ID"
    INVALID_AVATAR_ID = "INVALID_AVATAR_ID"
    INVALID_LANGUAGE_CODE = "INVALID_LANGUAGE_CODE"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"
    INVALID_QUALITY_LEVEL = "INVALID_QUALITY_LEVEL"
    
    # Cross-field validation
    SCRIPT_DURATION_MISMATCH = "SCRIPT_DURATION_MISMATCH"
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    TOTAL_SIZE_EXCEEDED = "TOTAL_SIZE_EXCEEDED"
    
    # Security validation
    SUSPICIOUS_CONTENT = "SUSPICIOUS_CONTENT"
    MALICIOUS_URL = "MALICIOUS_URL"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"


# =============================================================================
# Content Validation
# =============================================================================

def validate_script_content_v2(v: str) -> str:
    """Enhanced script content validation with comprehensive checks."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_SCRIPT_CONTENT,
            "Script must be a string"
        )
    
    # Strip whitespace
    v = v.strip()
    
    # Check for empty content
    if not v:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_SHORT,
            "Script cannot be empty"
        )
    
    # Check minimum length
    if len(v) < 10:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_SHORT,
            f"Script must be at least 10 characters long (current: {len(v)})"
        )
    
    # Check maximum length
    if len(v) > 10000:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_LONG,
            f"Script cannot exceed 10,000 characters (current: {len(v)})"
        )
    
    # Check for inappropriate content
    inappropriate_patterns = [
        r'\b(spam|scam|phishing|malware|virus)\b',
        r'\b(hack|crack|exploit|vulnerability)\b',
        r'\b(drugs|weapons|illegal)\b',
        r'\b(hate|discrimination|racism)\b'
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, v.lower()):
            raise PydanticCustomError(
                ValidationErrorCode.INAPPROPRIATE_CONTENT,
                "Script contains inappropriate content"
            )
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\b(click here|buy now|limited time|act fast)\b',
        r'\b(free money|get rich|earn money)\b',
        r'\b(password|credit card|social security)\b'
    ]
    
    suspicious_count = sum(1 for pattern in suspicious_patterns if re.search(pattern, v.lower()))
    if suspicious_count > 2:
        raise PydanticCustomError(
            ValidationErrorCode.SUSPICIOUS_CONTENT,
            "Script contains suspicious patterns"
        )
    
    # Check for balanced content (not too repetitive)
    words = v.lower().split()
    if len(words) > 10:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.3:  # More than 30% repetition
            raise PydanticCustomError(
                ValidationErrorCode.INAPPROPRIATE_CONTENT,
                "Script contains too much repetition"
            )
    
    return v


def validate_email_v2(v: str) -> str:
    """Enhanced email validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Email must be a string"
        )
    
    v = v.strip().lower()
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Invalid email format"
        )
    
    # Check for disposable email domains
    disposable_domains = {
        'tempmail.org', '10minutemail.com', 'guerrillamail.com',
        'mailinator.com', 'throwaway.email', 'temp-mail.org'
    }
    
    domain = v.split('@')[1]
    if domain in disposable_domains:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Disposable email addresses are not allowed"
        )
    
    # Check for suspicious patterns
    if re.search(r'(admin|root|test|demo|example)', v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Email contains suspicious patterns"
        )
    
    return v


def validate_url_v2(v: str) -> str:
    """Enhanced URL validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_URL_FORMAT,
            "URL must be a string"
        )
    
    v = v.strip()
    
    # Basic URL pattern
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_URL_FORMAT,
            "Invalid URL format"
        )
    
    # Parse URL
    try:
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise PydanticCustomError(
                ValidationErrorCode.INVALID_URL_FORMAT,
                "Invalid URL structure"
            )
    except Exception:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_URL_FORMAT,
            "Invalid URL format"
        )
    
    # Check for suspicious domains
    suspicious_domains = {
        'malware.com', 'phishing.com', 'scam.com', 'fake.com'
    }
    
    domain = parsed.netloc.lower()
    if domain in suspicious_domains:
        raise PydanticCustomError(
            ValidationErrorCode.MALICIOUS_URL,
            "URL contains suspicious domain"
        )
    
    # Check for file extensions
    file_extensions = {'.exe', '.bat', '.cmd', '.scr', '.pif'}
    if any(domain.endswith(ext) for ext in file_extensions):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_FILE_TYPE,
            "URL points to executable file"
        )
    
    return v


# =============================================================================
# Format Validation
# =============================================================================

def validate_id_format_v2(v: str, prefix: str = "") -> str:
    """Enhanced ID format validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_ID_FORMAT,
            f"{prefix}ID must be a string"
        )
    
    v = v.strip()
    
    # ID pattern: alphanumeric, hyphens, underscores, 3-50 characters
    id_pattern = r'^[a-zA-Z0-9_-]{3,50}$'
    if not re.match(id_pattern, v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_ID_FORMAT,
            f"{prefix}ID must be 3-50 characters long and contain only letters, numbers, hyphens, and underscores"
        )
    
    # Check for reserved patterns
    reserved_patterns = ['admin', 'root', 'system', 'test', 'demo']
    if v.lower() in reserved_patterns:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_ID_FORMAT,
            f"{prefix}ID cannot use reserved names"
        )
    
    return v


def validate_voice_id_v2(v: str) -> str:
    """Enhanced voice ID validation."""
    return validate_id_format_v2(v, "Voice ")


def validate_avatar_id_v2(v: str) -> str:
    """Enhanced avatar ID validation."""
    return validate_id_format_v2(v, "Avatar ")


def validate_phone_number_v2(v: str) -> str:
    """Enhanced phone number validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_PHONE_FORMAT,
            "Phone number must be a string"
        )
    
    v = v.strip()
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', v)
    
    # Check for valid phone number patterns
    if len(digits_only) < 10 or len(digits_only) > 15:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_PHONE_FORMAT,
            "Phone number must be 10-15 digits long"
        )
    
    # Check for suspicious patterns (all same digits, etc.)
    if len(set(digits_only)) == 1:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_PHONE_FORMAT,
            "Phone number cannot be all the same digit"
        )
    
    return v


# =============================================================================
# Range Validation
# =============================================================================

def validate_duration_v2(v: Union[int, float]) -> Union[int, float]:
    """Enhanced duration validation."""
    if not isinstance(v, (int, float)):
        raise PydanticCustomError(
            ValidationErrorCode.DURATION_OUT_OF_RANGE,
            "Duration must be a number"
        )
    
    if v < 5:
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_SMALL,
            "Duration must be at least 5 seconds"
        )
    
    if v > 3600:
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_LARGE,
            "Duration cannot exceed 1 hour (3600 seconds)"
        )
    
    return v


def validate_file_size_v2(v: int) -> int:
    """Enhanced file size validation."""
    if not isinstance(v, int):
        raise PydanticCustomError(
            ValidationErrorCode.FILE_SIZE_TOO_LARGE,
            "File size must be an integer"
        )
    
    if v < 0:
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_SMALL,
            "File size cannot be negative"
        )
    
    # 100MB limit
    max_size = 100 * 1024 * 1024
    if v > max_size:
        raise PydanticCustomError(
            ValidationErrorCode.FILE_SIZE_TOO_LARGE,
            f"File size cannot exceed {max_size // (1024 * 1024)}MB"
        )
    
    return v


def validate_percentage_v2(v: Union[int, float]) -> Union[int, float]:
    """Validate percentage values (0-100)."""
    if not isinstance(v, (int, float)):
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_SMALL,
            "Percentage must be a number"
        )
    
    if v < 0:
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_SMALL,
            "Percentage cannot be negative"
        )
    
    if v > 100:
        raise PydanticCustomError(
            ValidationErrorCode.VALUE_TOO_LARGE,
            "Percentage cannot exceed 100"
        )
    
    return v


# =============================================================================
# Business Logic Validation
# =============================================================================

def validate_language_code_v2(v: str) -> str:
    """Enhanced language code validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_LANGUAGE_CODE,
            "Language code must be a string"
        )
    
    v = v.lower().strip()
    
    # Supported language codes
    supported_languages = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ru', 'ar', 'hi'
    }
    
    if v not in supported_languages:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_LANGUAGE_CODE,
            f"Unsupported language code: {v}. Supported: {', '.join(sorted(supported_languages))}"
        )
    
    return v


def validate_resolution_v2(v: str) -> str:
    """Enhanced resolution validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_RESOLUTION,
            "Resolution must be a string"
        )
    
    v = v.lower().strip()
    
    # Supported resolutions
    supported_resolutions = {'720p', '1080p', '4k', '8k'}
    
    if v not in supported_resolutions:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_RESOLUTION,
            f"Unsupported resolution: {v}. Supported: {', '.join(sorted(supported_resolutions))}"
        )
    
    return v


def validate_quality_level_v2(v: str) -> str:
    """Enhanced quality level validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_QUALITY_LEVEL,
            "Quality level must be a string"
        )
    
    v = v.lower().strip()
    
    # Supported quality levels
    supported_qualities = {'low', 'medium', 'high', 'ultra'}
    
    if v not in supported_qualities:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_QUALITY_LEVEL,
            f"Unsupported quality level: {v}. Supported: {', '.join(sorted(supported_qualities))}"
        )
    
    return v


# =============================================================================
# Cross-Field Validation
# =============================================================================

def validate_script_duration_match(script: str, duration: Optional[int]) -> None:
    """Validate that script length matches duration."""
    if not duration:
        return
    
    # Estimate reading time (150 words per minute)
    word_count = len(script.split())
    estimated_minutes = word_count / 150
    
    if estimated_minutes > duration / 60:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_DURATION_MISMATCH,
            f"Script is too long for {duration} seconds. "
            f"Estimated reading time: {estimated_minutes:.1f} minutes"
        )


def validate_batch_limits(videos: List[Any]) -> None:
    """Validate batch processing limits."""
    if len(videos) > 20:
        raise PydanticCustomError(
            ValidationErrorCode.BATCH_SIZE_EXCEEDED,
            f"Batch size cannot exceed 20 videos (current: {len(videos)})"
        )
    
    # Calculate total estimated size
    total_size = sum(getattr(video, 'estimated_file_size', 0) for video in videos)
    max_total_size = 1024 * 1024 * 1024  # 1GB
    
    if total_size > max_total_size:
        raise PydanticCustomError(
            ValidationErrorCode.TOTAL_SIZE_EXCEEDED,
            f"Total estimated file size exceeds 1GB limit (current: {total_size // (1024 * 1024)}MB)"
        )


# =============================================================================
# Async Validators
# =============================================================================

async def validate_voice_exists_async(voice_id: str) -> str:
    """Async validator to check if voice exists in database."""
    # This would typically check against a database
    # For now, we'll simulate the check
    
    # Simulate database check
    await asyncio.sleep(0.01)  # Simulate database query
    
    # Mock validation - in real implementation, check database
    if voice_id.startswith('invalid_'):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_VOICE_ID,
            f"Voice ID '{voice_id}' does not exist"
        )
    
    return voice_id


async def validate_avatar_exists_async(avatar_id: str) -> str:
    """Async validator to check if avatar exists in database."""
    # Simulate database check
    await asyncio.sleep(0.01)
    
    # Mock validation
    if avatar_id.startswith('invalid_'):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_AVATAR_ID,
            f"Avatar ID '{avatar_id}' does not exist"
        )
    
    return avatar_id


async def validate_user_quota_async(user_id: str, operation: str) -> None:
    """Async validator to check user quota."""
    # Simulate quota check
    await asyncio.sleep(0.01)
    
    # Mock quota validation
    if user_id.startswith('quota_exceeded_'):
        raise PydanticCustomError(
            "QUOTA_EXCEEDED",
            f"User quota exceeded for operation: {operation}"
        )


# =============================================================================
# Custom Validator Decorators
# =============================================================================

def validate_content_safety(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate content safety."""
    def wrapper(*args, **kwargs) -> Any:
        # Extract content from arguments
        content = None
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10:
                content = arg
                break
        
        if not content:
            for value in kwargs.values():
                if isinstance(value, str) and len(value) > 10:
                    content = value
                    break
        
        if content:
            # Perform content safety check
            validate_script_content_v2(content)
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_rate_limit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate rate limits."""
    def wrapper(*args, **kwargs) -> Any:
        # This would typically check rate limits
        # For now, we'll just pass through
        return func(*args, **kwargs)
    
    return wrapper


# =============================================================================
# Performance Optimized Validators
# =============================================================================

class ValidationCache:
    """Cache for validation results to improve performance."""
    
    def __init__(self, max_size: int = 1000):
        
    """__init__ function."""
self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached validation result."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached validation result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# Global validation cache
validation_cache = ValidationCache()


def validate_with_cache(validator_func: Callable[[Any], Any], value: Any) -> bool:
    """Validate with caching for performance."""
    # Create cache key
    cache_key = f"{validator_func.__name__}:{hash(str(value))}"
    
    # Check cache
    cached_result = validation_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Perform validation
    result = validator_func(value)
    
    # Cache result
    validation_cache.set(cache_key, result)
    
    return result


# =============================================================================
# Batch Validation
# =============================================================================

def validate_batch_scripts(scripts: List[str]) -> List[str]:
    """Validate a batch of scripts efficiently."""
    validated_scripts = []
    errors = []
    
    for i, script in enumerate(scripts):
        try:
            validated_script = validate_script_content_v2(script)
            validated_scripts.append(validated_script)
        except PydanticCustomError as e:
            errors.append(f"Script {i + 1}: {e.message}")
    
    if errors:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_SCRIPT_CONTENT,
            f"Batch validation failed: {'; '.join(errors)}"
        )
    
    return validated_scripts


def validate_batch_emails(emails: List[str]) -> List[str]:
    """Validate a batch of emails efficiently."""
    validated_emails = []
    errors = []
    
    for i, email in enumerate(emails):
        try:
            validated_email = validate_email_v2(email)
            validated_emails.append(validated_email)
        except PydanticCustomError as e:
            errors.append(f"Email {i + 1}: {e.message}")
    
    if errors:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            f"Batch validation failed: {'; '.join(errors)}"
        )
    
    return validated_emails


# =============================================================================
# Custom Pydantic Validators
# =============================================================================

def create_custom_validator(
    validator_func: Callable[[Any], Any],
    error_code: str,
    error_message: str
) -> Callable[[Any], Any]:
    """Create a custom validator with error handling."""
    def wrapper(v: Any) -> Any:
        try:
            return validator_func(v)
        except Exception as e:
            raise PydanticCustomError(error_code, error_message)
    
    return wrapper


# =============================================================================
# Validation Utilities
# =============================================================================

def extract_validation_errors(validation_error: ValidationError) -> List[Dict[str, Any]]:
    """Extract validation errors in a structured format."""
    errors = []
    
    for error in validation_error.errors():
        error_info = {
            'field': '.'.join(str(loc) for loc in error['loc']),
            'message': error['msg'],
            'type': error['type'],
            'input': error.get('input'),
            'ctx': error.get('ctx', {})
        }
        errors.append(error_info)
    
    return errors


def format_validation_error(validation_error: ValidationError) -> str:
    """Format validation error as a user-friendly message."""
    errors = extract_validation_errors(validation_error)
    
    if len(errors) == 1:
        error = errors[0]
        return f"{error['field']}: {error['message']}"
    else:
        error_messages = [f"{error['field']}: {error['message']}" for error in errors]
        return f"Validation failed: {'; '.join(error_messages)}"


def validate_model_with_context(
    model_class: type,
    data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Validate model with additional context."""
    try:
        if context:
            # Add context to validation info
            validation_info = ValidationInfo(context=context)
            return model_class.model_validate(data, context=validation_info.context)
        else:
            return model_class.model_validate(data)
    except ValidationError as e:
        logger.error(
            "Model validation failed",
            model_class=model_class.__name__,
            errors=extract_validation_errors(e),
            context=context
        )
        raise


# =============================================================================
# Export all validators
# =============================================================================

__all__ = [
    # Error codes
    "ValidationErrorCode",
    
    # Content validation
    "validate_script_content_v2", "validate_email_v2", "validate_url_v2",
    
    # Format validation
    "validate_id_format_v2", "validate_voice_id_v2", "validate_avatar_id_v2", "validate_phone_number_v2",
    
    # Range validation
    "validate_duration_v2", "validate_file_size_v2", "validate_percentage_v2",
    
    # Business logic validation
    "validate_language_code_v2", "validate_resolution_v2", "validate_quality_level_v2",
    
    # Cross-field validation
    "validate_script_duration_match", "validate_batch_limits",
    
    # Async validators
    "validate_voice_exists_async", "validate_avatar_exists_async", "validate_user_quota_async",
    
    # Decorators
    "validate_content_safety", "validate_rate_limit",
    
    # Performance
    "ValidationCache", "validate_with_cache",
    
    # Batch validation
    "validate_batch_scripts", "validate_batch_emails",
    
    # Utilities
    "create_custom_validator", "extract_validation_errors", "format_validation_error", "validate_model_with_context",
] 