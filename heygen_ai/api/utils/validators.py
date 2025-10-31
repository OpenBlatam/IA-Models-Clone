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

import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from ..core.error_handling import (
from ..models.schemas import QualityLevel, LanguageCode, VideoStatus
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Validation utilities for HeyGen AI API
Provides comprehensive validation functions with early error handling and edge case management.
"""


    error_factory,
    validate_required,
    validate_length,
    validate_enum,
    validate_range,
    ValidationError,
    TimeoutError,
    ResourceExhaustionError,
    ConcurrencyError,
    ErrorLogger,
    UserFriendlyMessageGenerator
)

logger = logging.getLogger(__name__)

# Global validation state for resource tracking
_validation_locks = {}
_resource_usage = {}
_max_concurrent_validations = 10
_validation_semaphore = asyncio.Semaphore(_max_concurrent_validations)


def _acquire_validation_lock(resource_id: str) -> bool:
    """Acquire validation lock for resource to prevent concurrent validation conflicts"""
    if resource_id not in _validation_locks:
        _validation_locks[resource_id] = threading.Lock()
    
    return _validation_locks[resource_id].acquire(blocking=False)


def _release_validation_lock(resource_id: str) -> None:
    """Release validation lock for resource"""
    if resource_id not in _validation_locks:
        return
    
    try:
        _validation_locks[resource_id].release()
    except RuntimeError:
        # Lock was not acquired
        pass


def _check_resource_usage(resource_type: str, current_usage: float, limit: float) -> None:
    """Check resource usage and raise error if exceeded"""
    if current_usage <= limit:
        return
    
    raise error_factory.resource_exhaustion_error(
        message=f"{resource_type} usage exceeded limit",
        resource_type=resource_type,
        current_usage=current_usage,
        limit=limit
    )


def _validate_input_types(data: Dict[str, Any], expected_types: Dict[str, type]) -> None:
    """Validate input types at the beginning of functions"""
    for field, expected_type in expected_types.items():
        if field not in data:
            continue
        
        if isinstance(data[field], expected_type):
            continue
        
        raise error_factory.validation_error(
            message=f"Field '{field}' must be of type {expected_type.__name__}",
            field=field,
            value=data[field],
            validation_errors=[f"Expected {expected_type.__name__}, got {type(data[field]).__name__}"]
        )


def _validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate required fields at the beginning of functions"""
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
            continue
        
        if data[field] is None:
            missing_fields.append(field)
            continue
        
        if isinstance(data[field], str) and not data[field].strip():
            missing_fields.append(field)
            continue
    
    if not missing_fields:
        return
    
    raise error_factory.validation_error(
        message=f"Missing required fields: {', '.join(missing_fields)}",
        validation_errors=[f"Required fields: {', '.join(missing_fields)}"],
        details={"missing_fields": missing_fields}
    )


def _validate_string_safety(value: str, field_name: str) -> None:
    """Validate string safety to prevent injection attacks"""
    if not isinstance(value, str):
        return
    
    # Check for potential SQL injection patterns
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|\bOR\b|\bAND\b)",
        r"(\b(TRUE|FALSE|NULL)\b)",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise error_factory.validation_error(
                message=f"Field '{field_name}' contains potentially unsafe content",
                field=field_name,
                value="[REDACTED]",
                validation_errors=["Content contains potentially unsafe patterns"]
            )
    
    # Check for script injection patterns
    script_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise error_factory.validation_error(
                message=f"Field '{field_name}' contains potentially unsafe script content",
                field=field_name,
                value="[REDACTED]",
                validation_errors=["Content contains potentially unsafe script patterns"]
            )


def validate_video_id(video_id: str) -> bool:
    """Validate video ID format with early error handling"""
    # Early validation - check if video_id is provided
    if not video_id:
        return False
    
    # Early validation - check if video_id is string
    if not isinstance(video_id, str):
        return False
    
    # Early validation - check length
    if len(video_id) < 10 or len(video_id) > 100:
        return False
    
    # Video ID format: video_timestamp_userid_suffix
    pattern = re.compile(r'^video_\d+_[a-zA-Z0-9_-]+$')
    return bool(pattern.match(video_id))


def validate_script_content(script: str) -> Tuple[bool, List[str]]:
    """Validate script content for video generation with early error handling"""
    errors = []
    
    # Early validation - check if script is provided
    if not script:
        errors.append("Script cannot be empty")
        return False, errors
    
    # Early validation - check if script is string
    if not isinstance(script, str):
        errors.append("Script must be a string")
        return False, errors
    
    # Early validation - check for null bytes and control characters
    if '\x00' in script or any(ord(char) < 32 and char not in '\n\r\t' for char in script):
        errors.append("Script contains invalid control characters")
    
    # Early validation - check for encoding issues
    try:
        script.encode('utf-8')
    except UnicodeEncodeError:
        errors.append("Script contains invalid Unicode characters")
    
    # Check minimum length
    if len(script.strip()) < 10:
        errors.append("Script must be at least 10 characters long")
    
    # Check maximum length
    if len(script) > 5000:
        errors.append("Script cannot exceed 5000 characters")
    
    # Check for inappropriate content (basic check)
    inappropriate_words = ['spam', 'scam', 'inappropriate', 'malware', 'virus']
    script_lower = script.lower()
    for word in inappropriate_words:
        if word in script_lower:
            errors.append(f"Script contains inappropriate content: {word}")
    
    # Check for excessive repetition
    words = script.split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
        
        for word, count in word_counts.items():
            if count > len(words) * 0.3:  # More than 30% repetition
                errors.append(f"Excessive repetition of word: {word}")
    
    # Check for potential injection attacks
    try:
        _validate_string_safety(script, "script")
    except ValidationError as e:
        errors.extend(e.details.get('validation_errors', []))
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_voice_id(voice_id: str) -> Tuple[bool, List[str]]:
    """Validate voice ID for video generation with early error handling"""
    errors = []
    
    # Early validation - check if voice ID is provided
    if not voice_id:
        errors.append("Voice ID is required")
        return False, errors
    
    # Early validation - check if voice ID is string
    if not isinstance(voice_id, str):
        errors.append("Voice ID must be a string")
        return False, errors
    
    # Early validation - check length
    if len(voice_id) < 1 or len(voice_id) > 50:
        errors.append("Voice ID length must be between 1 and 50 characters")
        return False, errors
    
    # Valid voice IDs (in production, this would come from a database)
    valid_voices = [
        "Voice 1", "Voice 2", "Voice 3", "Voice 4", "Voice 5",
        "Male 1", "Male 2", "Female 1", "Female 2", "Neutral 1"
    ]
    
    if voice_id not in valid_voices:
        errors.append(f"Voice ID '{voice_id}' is not valid. Valid options: {', '.join(valid_voices)}")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_language_code(language: str) -> Tuple[bool, List[str]]:
    """Validate language code for video generation with early error handling"""
    errors = []
    
    # Early validation - check if language is provided
    if not language:
        errors.append("Language code is required")
        return False, errors
    
    # Early validation - check if language is string
    if not isinstance(language, str):
        errors.append("Language code must be a string")
        return False, errors
    
    # Early validation - check length
    if len(language) != 2:
        errors.append("Language code must be exactly 2 characters")
        return False, errors
    
    # Valid language codes
    valid_languages = [
        "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
        "ar", "hi", "nl", "sv", "no", "da", "fi", "pl", "tr", "th"
    ]
    
    if language not in valid_languages:
        errors.append(f"Language code '{language}' is not supported. Supported languages: {', '.join(valid_languages)}")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_quality_settings(quality: str, duration: Optional[int] = None) -> Tuple[bool, List[str]]:
    """Validate quality settings for video generation with early error handling"""
    errors = []
    
    # Early validation - check if quality is provided
    if not quality:
        errors.append("Quality level is required")
        return False, errors
    
    # Early validation - check if quality is string
    if not isinstance(quality, str):
        errors.append("Quality level must be a string")
        return False, errors
    
    # Valid quality levels
    valid_qualities = [QualityLevel.LOW.value, QualityLevel.MEDIUM.value, QualityLevel.HIGH.value]
    
    if quality not in valid_qualities:
        errors.append(f"Quality level '{quality}' is not valid. Valid options: {', '.join(valid_qualities)}")
        return False, errors
    
    # Check duration constraints based on quality
    if duration is None:
        return len(errors) == 0, errors
    
    if not isinstance(duration, int):
        errors.append("Duration must be an integer")
        return False, errors
    
    if quality == QualityLevel.LOW.value and duration > 120:
        errors.append("Low quality videos cannot exceed 120 seconds")
    
    if quality == QualityLevel.MEDIUM.value and duration > 300:
        errors.append("Medium quality videos cannot exceed 300 seconds")
    
    if quality == QualityLevel.HIGH.value and duration > 600:
        errors.append("High quality videos cannot exceed 600 seconds")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_video_duration(duration: Optional[int]) -> Tuple[bool, List[str]]:
    """Validate video duration with early error handling"""
    errors = []
    
    if duration is None:
        return True, errors
    
    # Early validation - check if duration is integer
    if not isinstance(duration, int):
        errors.append("Duration must be an integer")
        return False, errors
    
    # Early validation - check if duration is positive
    if duration <= 0:
        errors.append("Duration must be positive")
        return False, errors
    
    if duration < 5:
        errors.append("Video duration must be at least 5 seconds")
    
    if duration > 600:
        errors.append("Video duration cannot exceed 600 seconds (10 minutes)")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_user_id(user_id: str) -> Tuple[bool, List[str]]:
    """Validate user ID format with early error handling"""
    errors = []
    
    # Early validation - check if user_id is provided
    if not user_id:
        errors.append("User ID is required")
        return False, errors
    
    # Early validation - check if user_id is string
    if not isinstance(user_id, str):
        errors.append("User ID must be a string")
        return False, errors
    
    # Early validation - check length
    if len(user_id) < 3 or len(user_id) > 50:
        errors.append("User ID must be between 3 and 50 characters")
        return False, errors
    
    # User ID format validation (adjust based on your user ID format)
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        errors.append("User ID must contain only letters, numbers, underscores, and hyphens")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_email_format(email: str) -> Tuple[bool, List[str]]:
    """Validate email format with early error handling"""
    errors = []
    
    # Early validation - check if email is provided
    if not email:
        errors.append("Email is required")
        return False, errors
    
    # Early validation - check if email is string
    if not isinstance(email, str):
        errors.append("Email must be a string")
        return False, errors
    
    # Early validation - check length
    if len(email) > 254:  # RFC 5321 limit
        errors.append("Email address is too long")
        return False, errors
    
    # Email format validation
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    if not email_pattern.match(email):
        errors.append("Invalid email format")
    
    # Check for common disposable email domains
    disposable_domains = [
        'tempmail.org', '10minutemail.com', 'guerrillamail.com',
        'mailinator.com', 'yopmail.com', 'temp-mail.org'
    ]
    
    domain = email.split('@')[-1].lower()
    if domain in disposable_domains:
        errors.append("Disposable email addresses are not allowed")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_username_format(username: str) -> Tuple[bool, List[str]]:
    """Validate username format with early error handling"""
    errors = []
    
    # Early validation - check if username is provided
    if not username:
        errors.append("Username is required")
        return False, errors
    
    # Early validation - check if username is string
    if not isinstance(username, str):
        errors.append("Username must be a string")
        return False, errors
    
    # Username format validation
    if len(username) < 3:
        errors.append("Username must be at least 3 characters long")
    
    if len(username) > 30:
        errors.append("Username cannot exceed 30 characters")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        errors.append("Username can only contain letters, numbers, underscores, and hyphens")
    
    # Check for reserved usernames
    reserved_usernames = ['admin', 'root', 'system', 'support', 'help', 'info']
    if username.lower() in reserved_usernames:
        errors.append("This username is reserved and cannot be used")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """Validate password strength with early error handling"""
    errors = []
    
    # Early validation - check if password is provided
    if not password:
        errors.append("Password is required")
        return False, errors
    
    # Early validation - check if password is string
    if not isinstance(password, str):
        errors.append("Password must be a string")
        return False, errors
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if len(password) > 128:
        errors.append("Password cannot exceed 128 characters")
    
    # Check for required character types
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    # Check for common weak passwords
    weak_passwords = [
        'password', '123456', 'qwerty', 'admin', 'letmein',
        'welcome', 'monkey', 'dragon', 'master', 'football'
    ]
    
    if password.lower() in weak_passwords:
        errors.append("This password is too common and not secure")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_pagination_parameters(page: int, page_size: int) -> Tuple[bool, List[str]]:
    """Validate pagination parameters with early error handling"""
    errors = []
    
    # Early validation - check if parameters are integers
    if not isinstance(page, int):
        errors.append("Page number must be an integer")
        return False, errors
    
    if not isinstance(page_size, int):
        errors.append("Page size must be an integer")
        return False, errors
    
    if page < 1:
        errors.append("Page number must be greater than 0")
    
    if page_size < 1:
        errors.append("Page size must be greater than 0")
    
    if page_size > 100:
        errors.append("Page size cannot exceed 100")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_date_range(
    date_from: Optional[datetime],
    date_to: Optional[datetime],
    max_range_days: int = 365
) -> Tuple[bool, List[str]]:
    """Validate date range for analytics and filtering with early error handling"""
    errors = []
    
    if not date_from or not date_to:
        return True, errors
    
    # Early validation - check if dates are datetime objects
    if not isinstance(date_from, datetime):
        errors.append("Start date must be a datetime object")
        return False, errors
    
    if not isinstance(date_to, datetime):
        errors.append("End date must be a datetime object")
        return False, errors
    
    if date_from > date_to:
        errors.append("Start date cannot be after end date")
    
    date_diff = date_to - date_from
    if date_diff.days > max_range_days:
        errors.append(f"Date range cannot exceed {max_range_days} days")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


async def validate_api_key_format(api_key: str) -> Tuple[bool, List[str]]:
    """Validate API key format with early error handling"""
    errors = []
    
    # Early validation - check if API key is provided
    if not api_key:
        errors.append("API key is required")
        return False, errors
    
    # Early validation - check if API key is string
    if not isinstance(api_key, str):
        errors.append("API key must be a string")
        return False, errors
    
    # Early validation - check length
    if len(api_key) != 64:
        errors.append("API key must be exactly 64 characters long")
        return False, errors
    
    # API key format validation (64 character hex string)
    if not re.match(r'^[a-fA-F0-9]{64}$', api_key):
        errors.append("API key must be a 64-character hexadecimal string")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


def validate_rate_limit_parameters(
    requests_per_minute: int,
    burst_limit: int
) -> Tuple[bool, List[str]]:
    """Validate rate limiting parameters with early error handling"""
    errors = []
    
    # Early validation - check if parameters are integers
    if not isinstance(requests_per_minute, int):
        errors.append("Requests per minute must be an integer")
        return False, errors
    
    if not isinstance(burst_limit, int):
        errors.append("Burst limit must be an integer")
        return False, errors
    
    if requests_per_minute < 1:
        errors.append("Requests per minute must be at least 1")
    
    if requests_per_minute > 1000:
        errors.append("Requests per minute cannot exceed 1000")
    
    if burst_limit < 1:
        errors.append("Burst limit must be at least 1")
    
    if burst_limit > requests_per_minute * 2:
        errors.append("Burst limit cannot exceed twice the requests per minute")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


async def validate_video_processing_settings(settings: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate video processing settings with early error handling and resource checks"""
    errors = []
    
    # Early validation - check if settings is dict
    if not isinstance(settings, dict):
        errors.append("Settings must be a dictionary")
        return False, errors
    
    # Check resource usage before validation
    async with _validation_semaphore:
        _check_resource_usage("validation_slots", len(_resource_usage), _max_concurrent_validations)
        
        # Validate required fields
        required_fields = ['quality', 'voice_id', 'language']
        for field in required_fields:
            if field not in settings:
                errors.append(f"Required field '{field}' is missing")
        
        # Validate quality if present
        if 'quality' in settings:
            is_valid, quality_errors = validate_quality_settings(settings['quality'])
            if not is_valid:
                errors.extend(quality_errors)
        
        # Validate voice_id if present
        if 'voice_id' in settings:
            is_valid, voice_errors = validate_voice_id(settings['voice_id'])
            if not is_valid:
                errors.extend(voice_errors)
        
        # Validate language if present
        if 'language' in settings:
            is_valid, language_errors = validate_language_code(settings['language'])
            if not is_valid:
                errors.extend(language_errors)
        
        # Validate custom settings if present
        if 'custom_settings' in settings:
            custom_settings = settings['custom_settings']
            if not isinstance(custom_settings, dict):
                errors.append("Custom settings must be a dictionary")
            else:
                # Validate processing parameters
                if 'num_inference_steps' in custom_settings:
                    steps = custom_settings['num_inference_steps']
                    if not isinstance(steps, int) or steps < 1 or steps > 200:
                        errors.append("num_inference_steps must be an integer between 1 and 200")
                
                if 'guidance_scale' in custom_settings:
                    scale = custom_settings['guidance_scale']
                    if not isinstance(scale, (int, float)) or scale < 1.0 or scale > 20.0:
                        errors.append("guidance_scale must be a number between 1.0 and 20.0")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


async def validate_file_upload(
    file_size: int,
    file_type: str,
    max_size_mb: int = 100,
    allowed_types: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validate file upload parameters with early error handling"""
    errors = []
    
    # Early validation - check if file_size is integer
    if not isinstance(file_size, int):
        errors.append("File size must be an integer")
        return False, errors
    
    # Early validation - check if file_type is string
    if not isinstance(file_type, str):
        errors.append("File type must be a string")
        return False, errors
    
    # Validate file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        errors.append(f"File size cannot exceed {max_size_mb}MB")
    
    if file_size <= 0:
        errors.append("File size must be greater than 0")
    
    # Validate file type
    if allowed_types:
        if file_type.lower() not in [t.lower() for t in allowed_types]:
            errors.append(f"File type '{file_type}' is not allowed. Allowed types: {', '.join(allowed_types)}")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


async def validate_business_logic_constraints(
    user_id: str,
    operation: str,
    constraints: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate business logic constraints with early error handling and concurrency checks"""
    errors = []
    
    # Early validation - check if user_id is string
    if not isinstance(user_id, str):
        errors.append("User ID must be a string")
        return False, errors
    
    # Early validation - check if operation is string
    if not isinstance(operation, str):
        errors.append("Operation must be a string")
        return False, errors
    
    # Early validation - check if constraints is dict
    if not isinstance(constraints, dict):
        errors.append("Constraints must be a dictionary")
        return False, errors
    
    # Check for concurrent validation conflicts
    resource_id = f"validation_{user_id}_{operation}"
    if not _acquire_validation_lock(resource_id):
        raise error_factory.concurrency_error(
            message="Concurrent validation in progress",
            resource=resource_id,
            conflict_type="validation_lock"
        )
    
    try:
        # Example business logic validations
        if operation == "video_generation":
            # Check daily video generation limit
            daily_limit = constraints.get('daily_video_limit', 10)
            # In production, you would check the actual count from database
            # current_count = get_user_daily_video_count(user_id)
            # if current_count >= daily_limit:
            #     errors.append(f"Daily video generation limit of {daily_limit} reached")
            
            # Check user subscription status
            subscription_status = constraints.get('subscription_status', 'active')
            if subscription_status != 'active':
                errors.append("Active subscription required for video generation")
        
        if operation == "api_usage":
            # Check API usage limits
            api_limit = constraints.get('api_limit', 1000)
            # In production, you would check the actual usage from database
            # current_usage = get_user_api_usage(user_id)
            # if current_usage >= api_limit:
            #     errors.append(f"API usage limit of {api_limit} reached")
            pass
    
    finally:
        _release_validation_lock(resource_id)
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors


# Enhanced convenience functions with early error handling
async def validate_video_generation_request(data: Dict[str, Any]) -> None:
    """Validate complete video generation request with early error handling"""
    # Early validation - check if data is dict
    if not isinstance(data, dict):
        raise error_factory.validation_error(
            message="Request data must be a dictionary",
            validation_errors=["Invalid data type"]
        )
    
    # Early validation - check required fields
    required_fields = ["script", "voice_id", "language", "quality"]
    _validate_required_fields(data, required_fields)
    
    # Early validation - check field types
    expected_types = {
        "script": str,
        "voice_id": str,
        "language": str,
        "quality": str,
        "duration": (int, type(None))
    }
    _validate_input_types(data, expected_types)
    
    # Validate script content
    script = data.get("script", "")
    is_valid, errors = validate_script_content(script)
    if not is_valid:
        raise error_factory.validation_error(
            message="Script validation failed",
            field="script",
            value=script,
            validation_errors=errors
        )
    
    # Validate voice ID
    voice_id = data.get("voice_id", "")
    is_valid, errors = validate_voice_id(voice_id)
    if not is_valid:
        raise error_factory.validation_error(
            message="Voice ID validation failed",
            field="voice_id",
            value=voice_id,
            validation_errors=errors
        )
    
    # Validate language
    language = data.get("language", "")
    is_valid, errors = validate_language_code(language)
    if not is_valid:
        raise error_factory.validation_error(
            message="Language validation failed",
            field="language",
            value=language,
            validation_errors=errors
        )
    
    # Validate quality settings
    quality = data.get("quality", "")
    duration = data.get("duration")
    is_valid, errors = validate_quality_settings(quality, duration)
    if not is_valid:
        raise error_factory.validation_error(
            message="Quality settings validation failed",
            field="quality",
            value=quality,
            validation_errors=errors
        )
    
    # Validate duration if provided
    if duration is not None:
        is_valid, errors = validate_video_duration(duration)
        if not is_valid:
            raise error_factory.validation_error(
                message="Duration validation failed",
                field="duration",
                value=duration,
                validation_errors=errors
            )


def validate_user_registration_data(data: Dict[str, Any]) -> None:
    """Validate user registration data with early error handling"""
    # Early validation - check if data is dict
    if not isinstance(data, dict):
        raise error_factory.validation_error(
            message="Registration data must be a dictionary",
            validation_errors=["Invalid data type"]
        )
    
    # Early validation - check required fields
    required_fields = ["username", "email", "password"]
    _validate_required_fields(data, required_fields)
    
    # Early validation - check field types
    expected_types = {
        "username": str,
        "email": str,
        "password": str
    }
    _validate_input_types(data, expected_types)
    
    # Validate username format
    username = data.get("username", "")
    is_valid, errors = validate_username_format(username)
    if not is_valid:
        raise error_factory.validation_error(
            message="Username validation failed",
            field="username",
            value=username,
            validation_errors=errors
        )
    
    # Validate email format
    email = data.get("email", "")
    is_valid, errors = validate_email_format(email)
    if not is_valid:
        raise error_factory.validation_error(
            message="Email validation failed",
            field="email",
            value=email,
            validation_errors=errors
        )
    
    # Validate password strength
    password = data.get("password", "")
    is_valid, errors = validate_password_strength(password)
    if not is_valid:
        raise error_factory.validation_error(
            message="Password validation failed",
            field="password",
            value="[REDACTED]",
            validation_errors=errors
        )


# Named exports
__all__ = [
    "validate_video_id",
    "validate_script_content",
    "validate_voice_id",
    "validate_language_code",
    "validate_quality_settings",
    "validate_video_duration",
    "validate_user_id",
    "validate_email_format",
    "validate_username_format",
    "validate_password_strength",
    "validate_pagination_parameters",
    "validate_date_range",
    "validate_api_key_format",
    "validate_rate_limit_parameters",
    "validate_video_processing_settings",
    "validate_file_upload",
    "validate_business_logic_constraints",
    "validate_video_generation_request",
    "validate_user_registration_data"
] 