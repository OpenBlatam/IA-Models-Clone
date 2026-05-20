from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, Optional, List, Union, Callable, Type, Tuple
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from functools import wraps
import logging
import time
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import traceback
import hashlib
import json
import uuid
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Error Handling System for HeyGen AI API
Provides comprehensive error handling, validation, and edge case management.
"""


logger = logging.getLogger(__name__)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create structured logger for errors
error_logger = logging.getLogger('heygen.errors')
error_logger.setLevel(logging.ERROR)

# Create structured logger for user actions
user_logger = logging.getLogger('heygen.user_actions')
user_logger.setLevel(logging.INFO)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERIALIZATION = "serialization"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    VIDEO_PROCESSING = "video_processing"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_EXHAUSTED = "retry_exhausted"
    CONCURRENCY = "concurrency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    # New specific error categories
    VIDEO_GENERATION = "video_generation"
    VIDEO_RENDERING = "video_rendering"
    VIDEO_UPLOAD = "video_upload"
    VIDEO_DOWNLOAD = "video_download"
    TEMPLATE_PROCESSING = "template_processing"
    VOICE_SYNTHESIS = "voice_synthesis"
    AUDIO_PROCESSING = "audio_processing"
    FILE_PROCESSING = "file_processing"
    API_INTEGRATION = "api_integration"
    PAYMENT_PROCESSING = "payment_processing"
    USER_MANAGEMENT = "user_management"
    CONTENT_MODERATION = "content_moderation"
    QUOTA_EXCEEDED = "quota_exceeded"
    FEATURE_UNAVAILABLE = "feature_unavailable"
    MAINTENANCE_MODE = "maintenance_mode"
    DEPRECATED_FEATURE = "deprecated_feature"


class ErrorLogger:
    """Enhanced error logging with structured data and user-friendly messages"""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """Log error with structured data and context"""
        error_data = {
            'error_id': getattr(error, 'error_id', str(uuid.uuid4())),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'request_id': request_id,
            'operation': operation,
            'context': context or {},
            'severity': getattr(error, 'severity', ErrorSeverity.MEDIUM).value,
            'category': getattr(error, 'category', ErrorCategory.SYSTEM).value,
            'error_code': getattr(error, 'error_code', 'UNKNOWN_ERROR'),
            'details': getattr(error, 'details', {}),
            'stack_trace': traceback.format_exc() if hasattr(error, 'severity') else None
        }
        
        # Log based on severity
        severity = getattr(error, 'severity', ErrorSeverity.MEDIUM)
        if severity == ErrorSeverity.CRITICAL:
            error_logger.critical(json.dumps(error_data, default=str))
        elif severity == ErrorSeverity.HIGH:
            error_logger.error(json.dumps(error_data, default=str))
        elif severity == ErrorSeverity.MEDIUM:
            error_logger.warning(json.dumps(error_data, default=str))
        else:
            error_logger.info(json.dumps(error_data, default=str))
    
    @staticmethod
    def log_user_action(
        action: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """Log user actions for audit and debugging"""
        action_data = {
            'action': action,
            'user_id': user_id,
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'success': success,
            'details': details or {}
        }
        
        if success:
            user_logger.info(json.dumps(action_data, default=str))
        else:
            user_logger.warning(json.dumps(action_data, default=str))


class UserFriendlyMessageGenerator:
    """Generate user-friendly error messages"""
    
    @staticmethod
    def get_validation_message(field: Optional[str] = None, error_type: str = "validation") -> str:
        """Generate user-friendly validation error messages"""
        if field:
            return f"Please check the '{field}' field and try again."
        
        messages = {
            "validation": "Please check your input and try again.",
            "required": "This field is required. Please provide a value.",
            "format": "The format is incorrect. Please check and try again.",
            "length": "The length is not valid. Please adjust and try again.",
            "type": "The data type is incorrect. Please check and try again."
        }
        return messages.get(error_type, "Please check your input and try again.")
    
    @staticmethod
    def get_authentication_message() -> str:
        """Generate user-friendly authentication error messages"""
        return "Please log in again to continue."
    
    @staticmethod
    def get_authorization_message() -> str:
        """Generate user-friendly authorization error messages"""
        return "You don't have permission to perform this action."
    
    @staticmethod
    def get_resource_not_found_message(resource_type: Optional[str] = None) -> str:
        """Generate user-friendly resource not found messages"""
        if resource_type:
            return f"The {resource_type} you're looking for doesn't exist."
        return "The requested resource was not found."
    
    @staticmethod
    def get_rate_limit_message(retry_after: Optional[int] = None) -> str:
        """Generate user-friendly rate limit messages"""
        if retry_after:
            return f"Too many requests. Please try again in {retry_after} seconds."
        return "Too many requests. Please try again later."
    
    @staticmethod
    def get_timeout_message() -> str:
        """Generate user-friendly timeout messages"""
        return "The request took too long to process. Please try again."
    
    @staticmethod
    def get_video_processing_message(stage: Optional[str] = None) -> str:
        """Generate user-friendly video processing messages"""
        if stage:
            return f"Video processing failed at {stage}. Please try again."
        return "Video processing failed. Please try again."
    
    @staticmethod
    def get_system_error_message() -> str:
        """Generate user-friendly system error messages"""
        return "Something went wrong on our end. Please try again later."
    
    @staticmethod
    def get_database_error_message() -> str:
        """Generate user-friendly database error messages"""
        return "We're experiencing technical difficulties. Please try again later."
    
    # New domain-specific message generators
    @staticmethod
    def get_video_generation_message(stage: Optional[str] = None) -> str:
        """Generate user-friendly video generation messages"""
        if stage:
            return f"Video generation failed at {stage}. Please try again."
        return "Video generation failed. Please try again."
    
    @staticmethod
    def get_video_rendering_message(stage: Optional[str] = None) -> str:
        """Generate user-friendly video rendering messages"""
        if stage:
            return f"Video rendering failed at {stage}. Please try again."
        return "Video rendering failed. Please try again."
    
    @staticmethod
    def get_voice_synthesis_message(language: Optional[str] = None) -> str:
        """Generate user-friendly voice synthesis messages"""
        if language:
            return f"Voice synthesis failed for {language}. Please try a different voice or language."
        return "Voice synthesis failed. Please try a different voice."
    
    @staticmethod
    def get_template_processing_message(template_type: Optional[str] = None) -> str:
        """Generate user-friendly template processing messages"""
        if template_type:
            return f"Template processing failed for {template_type}. Please try a different template."
        return "Template processing failed. Please try a different template."
    
    @staticmethod
    def get_file_processing_message(operation: Optional[str] = None) -> str:
        """Generate user-friendly file processing messages"""
        if operation:
            return f"File {operation} failed. Please try again."
        return "File processing failed. Please try again."
    
    @staticmethod
    def get_quota_exceeded_message(quota_type: Optional[str] = None, reset_time: Optional[datetime] = None) -> str:
        """Generate user-friendly quota exceeded messages"""
        if quota_type and reset_time:
            return f"Your {quota_type} quota has been exceeded. It will reset at {reset_time.strftime('%H:%M')}."
        elif quota_type:
            return f"Your {quota_type} quota has been exceeded. Please try again later."
        return "Your quota has been exceeded. Please try again later."
    
    @staticmethod
    def get_content_moderation_message(content_type: Optional[str] = None) -> str:
        """Generate user-friendly content moderation messages"""
        if content_type:
            return f"Your {content_type} content was flagged. Please review and modify it."
        return "Your content was flagged. Please review and modify it."
    
    @staticmethod
    def get_feature_unavailable_message(feature_name: Optional[str] = None, reason: Optional[str] = None) -> str:
        """Generate user-friendly feature unavailable messages"""
        if feature_name and reason:
            return f"The {feature_name} feature is currently unavailable: {reason}."
        elif feature_name:
            return f"The {feature_name} feature is currently unavailable."
        return "This feature is currently unavailable."
    
    @staticmethod
    def get_maintenance_mode_message(estimated_duration: Optional[int] = None) -> str:
        """Generate user-friendly maintenance mode messages"""
        if estimated_duration:
            return f"We're currently performing maintenance. Please try again in {estimated_duration} minutes."
        return "We're currently performing maintenance. Please try again later."
    
    @staticmethod
    def get_deprecated_feature_message(feature_name: Optional[str] = None, replacement_feature: Optional[str] = None) -> str:
        """Generate user-friendly deprecated feature messages"""
        if feature_name and replacement_feature:
            return f"The {feature_name} feature has been deprecated. Please use {replacement_feature} instead."
        elif feature_name:
            return f"The {feature_name} feature has been deprecated."
        return "This feature has been deprecated."


class HeyGenBaseError(Exception):
    """Base exception for all HeyGen AI errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_friendly_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
        max_retries: Optional[int] = None,
        circuit_breaker_state: Optional[str] = None
    ):
        
    """__init__ function."""
self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.user_friendly_message = user_friendly_message or self._generate_user_friendly_message()
        self.details = details or {}
        self.context = context or {}
        self.retry_after = retry_after
        self.max_retries = max_retries
        self.circuit_breaker_state = circuit_breaker_state
        self.timestamp = datetime.utcnow()
        self.error_id = self._generate_error_id()
        super().__init__(self.message)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        error_data = f"{self.error_code}_{self.timestamp.isoformat()}_{self.message}"
        return hashlib.md5(error_data.encode()).hexdigest()[:8]
    
    def _generate_user_friendly_message(self) -> str:
        """Generate user-friendly message based on error category"""
        if self.category == ErrorCategory.VALIDATION:
            field = self.details.get('field')
            return UserFriendlyMessageGenerator.get_validation_message(field)
        elif self.category == ErrorCategory.AUTHENTICATION:
            return UserFriendlyMessageGenerator.get_authentication_message()
        elif self.category == ErrorCategory.AUTHORIZATION:
            return UserFriendlyMessageGenerator.get_authorization_message()
        elif self.category == ErrorCategory.RESOURCE_NOT_FOUND:
            resource_type = self.details.get('resource_type')
            return UserFriendlyMessageGenerator.get_resource_not_found_message(resource_type)
        elif self.category == ErrorCategory.RATE_LIMIT:
            return UserFriendlyMessageGenerator.get_rate_limit_message(self.retry_after)
        elif self.category == ErrorCategory.TIMEOUT:
            return UserFriendlyMessageGenerator.get_timeout_message()
        elif self.category == ErrorCategory.VIDEO_PROCESSING:
            stage = self.details.get('processing_stage')
            return UserFriendlyMessageGenerator.get_video_processing_message(stage)
        elif self.category == ErrorCategory.DATABASE:
            return UserFriendlyMessageGenerator.get_database_error_message()
        # New error categories
        elif self.category == ErrorCategory.VIDEO_GENERATION:
            stage = self.details.get('generation_stage')
            return UserFriendlyMessageGenerator.get_video_generation_message(stage)
        elif self.category == ErrorCategory.VIDEO_RENDERING:
            stage = self.details.get('rendering_stage')
            return UserFriendlyMessageGenerator.get_video_rendering_message(stage)
        elif self.category == ErrorCategory.VOICE_SYNTHESIS:
            language = self.details.get('language')
            return UserFriendlyMessageGenerator.get_voice_synthesis_message(language)
        elif self.category == ErrorCategory.TEMPLATE_PROCESSING:
            template_type = self.details.get('template_type')
            return UserFriendlyMessageGenerator.get_template_processing_message(template_type)
        elif self.category == ErrorCategory.FILE_PROCESSING:
            operation = self.details.get('operation')
            return UserFriendlyMessageGenerator.get_file_processing_message(operation)
        elif self.category == ErrorCategory.QUOTA_EXCEEDED:
            quota_type = self.details.get('quota_type')
            reset_time_str = self.details.get('reset_time')
            reset_time = None
            if reset_time_str:
                try:
                    reset_time = datetime.fromisoformat(reset_time_str)
                except ValueError:
                    pass
            return UserFriendlyMessageGenerator.get_quota_exceeded_message(quota_type, reset_time)
        elif self.category == ErrorCategory.CONTENT_MODERATION:
            content_type = self.details.get('content_type')
            return UserFriendlyMessageGenerator.get_content_moderation_message(content_type)
        elif self.category == ErrorCategory.FEATURE_UNAVAILABLE:
            feature_name = self.details.get('feature_name')
            reason = self.details.get('reason')
            return UserFriendlyMessageGenerator.get_feature_unavailable_message(feature_name, reason)
        elif self.category == ErrorCategory.MAINTENANCE_MODE:
            estimated_duration = self.details.get('estimated_duration')
            return UserFriendlyMessageGenerator.get_maintenance_mode_message(estimated_duration)
        elif self.category == ErrorCategory.DEPRECATED_FEATURE:
            feature_name = self.details.get('feature_name')
            replacement_feature = self.details.get('replacement_feature')
            return UserFriendlyMessageGenerator.get_deprecated_feature_message(feature_name, replacement_feature)
        else:
            return UserFriendlyMessageGenerator.get_system_error_message()


class ValidationError(HeyGenBaseError):
    """Validation error for input data"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        if validation_errors:
            details['validation_errors'] = validation_errors
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            **kwargs
        )


class AuthenticationError(HeyGenBaseError):
    """Authentication error"""
    
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(HeyGenBaseError):
    """Authorization error"""
    
    def __init__(self, message: str, required_permissions: Optional[List[str]] = None, **kwargs):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if required_permissions:
            details['required_permissions'] = required_permissions
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class DatabaseError(HeyGenBaseError):
    """Database operation error"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class ResourceNotFoundError(HeyGenBaseError):
    """Resource not found error"""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class VideoProcessingError(HeyGenBaseError):
    """Video processing error"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        if processing_stage:
            details['processing_stage'] = processing_stage
        
        super().__init__(
            message=message,
            error_code="VIDEO_PROCESSING_ERROR",
            category=ErrorCategory.VIDEO_PROCESSING,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class RateLimitError(HeyGenBaseError):
    """Rate limit error"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if limit:
            details['limit'] = limit
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            details=details,
            **kwargs
        )


class ExternalServiceError(HeyGenBaseError):
    """External service error"""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_response: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if service_response:
            details['service_response'] = service_response
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class TimeoutError(HeyGenBaseError):
    """Timeout error"""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if timeout_duration:
            details['timeout_duration'] = timeout_duration
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class CircuitBreakerError(HeyGenBaseError):
    """Circuit breaker error"""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        failure_count: Optional[int] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if failure_count:
            details['failure_count'] = failure_count
        
        super().__init__(
            message=message,
            error_code="CIRCUIT_BREAKER_ERROR",
            category=ErrorCategory.CIRCUIT_BREAKER,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class RetryExhaustedError(HeyGenBaseError):
    """Retry exhausted error"""
    
    def __init__(
        self,
        message: str,
        max_retries: int,
        attempts_made: int,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        details['max_retries'] = max_retries
        details['attempts_made'] = attempts_made
        
        super().__init__(
            message=message,
            error_code="RETRY_EXHAUSTED_ERROR",
            category=ErrorCategory.RETRY_EXHAUSTED,
            severity=ErrorSeverity.HIGH,
            max_retries=max_retries,
            details=details,
            **kwargs
        )


class ConcurrencyError(HeyGenBaseError):
    """Concurrency error"""
    
    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        conflict_type: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if resource:
            details['resource'] = resource
        if conflict_type:
            details['conflict_type'] = conflict_type
        
        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            category=ErrorCategory.CONCURRENCY,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ResourceExhaustionError(HeyGenBaseError):
    """Resource exhaustion error"""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if current_usage:
            details['current_usage'] = current_usage
        if limit:
            details['limit'] = limit
        
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTION_ERROR",
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


# New specific custom error types
class VideoGenerationError(HeyGenBaseError):
    """Video generation specific error"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        generation_stage: Optional[str] = None,
        template_id: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        if generation_stage:
            details['generation_stage'] = generation_stage
        if template_id:
            details['template_id'] = template_id
        
        super().__init__(
            message=message,
            error_code="VIDEO_GENERATION_ERROR",
            category=ErrorCategory.VIDEO_GENERATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class VideoRenderingError(HeyGenBaseError):
    """Video rendering specific error"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        rendering_stage: Optional[str] = None,
        render_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        if rendering_stage:
            details['rendering_stage'] = rendering_stage
        if render_settings:
            details['render_settings'] = render_settings
        
        super().__init__(
            message=message,
            error_code="VIDEO_RENDERING_ERROR",
            category=ErrorCategory.VIDEO_RENDERING,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class VoiceSynthesisError(HeyGenBaseError):
    """Voice synthesis specific error"""
    
    def __init__(
        self,
        message: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if voice_id:
            details['voice_id'] = voice_id
        if language:
            details['language'] = language
        if text_length:
            details['text_length'] = text_length
        
        super().__init__(
            message=message,
            error_code="VOICE_SYNTHESIS_ERROR",
            category=ErrorCategory.VOICE_SYNTHESIS,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class TemplateProcessingError(HeyGenBaseError):
    """Template processing specific error"""
    
    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
        template_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if template_id:
            details['template_id'] = template_id
        if template_type:
            details['template_type'] = template_type
        if processing_stage:
            details['processing_stage'] = processing_stage
        
        super().__init__(
            message=message,
            error_code="TEMPLATE_PROCESSING_ERROR",
            category=ErrorCategory.TEMPLATE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class FileProcessingError(HeyGenBaseError):
    """File processing specific error"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if file_type:
            details['file_type'] = file_type
        if file_size:
            details['file_size'] = file_size
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class QuotaExceededError(HeyGenBaseError):
    """Quota exceeded error"""
    
    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if quota_type:
            details['quota_type'] = quota_type
        if current_usage:
            details['current_usage'] = current_usage
        if limit:
            details['limit'] = limit
        if reset_time:
            details['reset_time'] = reset_time.isoformat()
        
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED_ERROR",
            category=ErrorCategory.QUOTA_EXCEEDED,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ContentModerationError(HeyGenBaseError):
    """Content moderation error"""
    
    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        moderation_result: Optional[str] = None,
        flagged_content: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if content_type:
            details['content_type'] = content_type
        if moderation_result:
            details['moderation_result'] = moderation_result
        if flagged_content:
            details['flagged_content'] = flagged_content
        
        super().__init__(
            message=message,
            error_code="CONTENT_MODERATION_ERROR",
            category=ErrorCategory.CONTENT_MODERATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class FeatureUnavailableError(HeyGenBaseError):
    """Feature unavailable error"""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        reason: Optional[str] = None,
        available_alternatives: Optional[List[str]] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if feature_name:
            details['feature_name'] = feature_name
        if reason:
            details['reason'] = reason
        if available_alternatives:
            details['available_alternatives'] = available_alternatives
        
        super().__init__(
            message=message,
            error_code="FEATURE_UNAVAILABLE_ERROR",
            category=ErrorCategory.FEATURE_UNAVAILABLE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class MaintenanceModeError(HeyGenBaseError):
    """Maintenance mode error"""
    
    def __init__(
        self,
        message: str,
        maintenance_start: Optional[datetime] = None,
        estimated_duration: Optional[int] = None,
        affected_services: Optional[List[str]] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if maintenance_start:
            details['maintenance_start'] = maintenance_start.isoformat()
        if estimated_duration:
            details['estimated_duration'] = estimated_duration
        if affected_services:
            details['affected_services'] = affected_services
        
        super().__init__(
            message=message,
            error_code="MAINTENANCE_MODE_ERROR",
            category=ErrorCategory.MAINTENANCE_MODE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class DeprecatedFeatureError(HeyGenBaseError):
    """Deprecated feature error"""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        deprecation_date: Optional[datetime] = None,
        replacement_feature: Optional[str] = None,
        migration_guide: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
details = kwargs.get('details', {})
        if feature_name:
            details['feature_name'] = feature_name
        if deprecation_date:
            details['deprecation_date'] = deprecation_date.isoformat()
        if replacement_feature:
            details['replacement_feature'] = replacement_feature
        if migration_guide:
            details['migration_guide'] = migration_guide
        
        super().__init__(
            message=message,
            error_code="DEPRECATED_FEATURE_ERROR",
            category=ErrorCategory.DEPRECATED_FEATURE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ErrorFactory:
    """Factory for creating error instances with proper logging"""
    
    @staticmethod
    def validation_error(
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ) -> ValidationError:
        """Create validation error with logging"""
        error = ValidationError(
            message=message,
            field=field,
            value=value,
            validation_errors=validation_errors,
            **kwargs
        )
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def authentication_error(message: str, **kwargs) -> AuthenticationError:
        """Create authentication error with logging"""
        error = AuthenticationError(message, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def authorization_error(
        message: str,
        required_permissions: Optional[List[str]] = None,
        **kwargs
    ) -> AuthorizationError:
        """Create authorization error with logging"""
        error = AuthorizationError(message, required_permissions, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def database_error(
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ) -> DatabaseError:
        """Create database error with logging"""
        error = DatabaseError(message, operation, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def resource_not_found_error(
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> ResourceNotFoundError:
        """Create resource not found error with logging"""
        error = ResourceNotFoundError(message, resource_type, resource_id, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def video_processing_error(
        message: str,
        video_id: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ) -> VideoProcessingError:
        """Create video processing error with logging"""
        error = VideoProcessingError(message, video_id, processing_stage, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def rate_limit_error(
        message: str,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> RateLimitError:
        """Create rate limit error with logging"""
        error = RateLimitError(message, retry_after, limit, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def external_service_error(
        message: str,
        service_name: Optional[str] = None,
        service_response: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ExternalServiceError:
        """Create external service error with logging"""
        error = ExternalServiceError(message, service_name, service_response, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def timeout_error(
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ) -> TimeoutError:
        """Create timeout error with logging"""
        error = TimeoutError(message, timeout_duration, operation, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def circuit_breaker_error(
        message: str,
        service_name: Optional[str] = None,
        failure_count: Optional[int] = None,
        **kwargs
    ) -> CircuitBreakerError:
        """Create circuit breaker error with logging"""
        error = CircuitBreakerError(message, service_name, failure_count, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def retry_exhausted_error(
        message: str,
        max_retries: int,
        attempts_made: int,
        **kwargs
    ) -> RetryExhaustedError:
        """Create retry exhausted error with logging"""
        error = RetryExhaustedError(message, max_retries, attempts_made, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def concurrency_error(
        message: str,
        resource: Optional[str] = None,
        conflict_type: Optional[str] = None,
        **kwargs
    ) -> ConcurrencyError:
        """Create concurrency error with logging"""
        error = ConcurrencyError(message, resource, conflict_type, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def resource_exhaustion_error(
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ) -> ResourceExhaustionError:
        """Create resource exhaustion error with logging"""
        error = ResourceExhaustionError(message, resource_type, current_usage, limit, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    # New domain-specific error creation methods
    @staticmethod
    def video_generation_error(
        message: str,
        video_id: Optional[str] = None,
        generation_stage: Optional[str] = None,
        template_id: Optional[str] = None,
        **kwargs
    ) -> VideoGenerationError:
        """Create video generation error with logging"""
        error = VideoGenerationError(message, video_id, generation_stage, template_id, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def video_rendering_error(
        message: str,
        video_id: Optional[str] = None,
        rendering_stage: Optional[str] = None,
        render_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VideoRenderingError:
        """Create video rendering error with logging"""
        error = VideoRenderingError(message, video_id, rendering_stage, render_settings, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def voice_synthesis_error(
        message: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs
    ) -> VoiceSynthesisError:
        """Create voice synthesis error with logging"""
        error = VoiceSynthesisError(message, voice_id, language, text_length, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def template_processing_error(
        message: str,
        template_id: Optional[str] = None,
        template_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ) -> TemplateProcessingError:
        """Create template processing error with logging"""
        error = TemplateProcessingError(message, template_id, template_type, processing_stage, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def file_processing_error(
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ) -> FileProcessingError:
        """Create file processing error with logging"""
        error = FileProcessingError(message, file_path, file_type, file_size, operation, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def quota_exceeded_error(
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ) -> QuotaExceededError:
        """Create quota exceeded error with logging"""
        error = QuotaExceededError(message, quota_type, current_usage, limit, reset_time, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def content_moderation_error(
        message: str,
        content_type: Optional[str] = None,
        moderation_result: Optional[str] = None,
        flagged_content: Optional[str] = None,
        **kwargs
    ) -> ContentModerationError:
        """Create content moderation error with logging"""
        error = ContentModerationError(message, content_type, moderation_result, flagged_content, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def feature_unavailable_error(
        message: str,
        feature_name: Optional[str] = None,
        reason: Optional[str] = None,
        available_alternatives: Optional[List[str]] = None,
        **kwargs
    ) -> FeatureUnavailableError:
        """Create feature unavailable error with logging"""
        error = FeatureUnavailableError(message, feature_name, reason, available_alternatives, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def maintenance_mode_error(
        message: str,
        maintenance_start: Optional[datetime] = None,
        estimated_duration: Optional[int] = None,
        affected_services: Optional[List[str]] = None,
        **kwargs
    ) -> MaintenanceModeError:
        """Create maintenance mode error with logging"""
        error = MaintenanceModeError(message, maintenance_start, estimated_duration, affected_services, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    @staticmethod
    def deprecated_feature_error(
        message: str,
        feature_name: Optional[str] = None,
        deprecation_date: Optional[datetime] = None,
        replacement_feature: Optional[str] = None,
        migration_guide: Optional[str] = None,
        **kwargs
    ) -> DeprecatedFeatureError:
        """Create deprecated feature error with logging"""
        error = DeprecatedFeatureError(message, feature_name, deprecation_date, replacement_feature, migration_guide, **kwargs)
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
    
    # Convenience methods for common error scenarios
    @staticmethod
    def invalid_video_id_error(video_id: str, **kwargs) -> ValidationError:
        """Create invalid video ID error"""
        return error_factory.validation_error(
            message=f"Invalid video ID format: {video_id}",
            field="video_id",
            value=video_id,
            validation_errors=["Video ID must match expected format"],
            **kwargs
        )
    
    @staticmethod
    def video_not_found_error(video_id: str, **kwargs) -> ResourceNotFoundError:
        """Create video not found error"""
        return error_factory.resource_not_found_error(
            message=f"Video with ID '{video_id}' not found",
            resource_type="video",
            resource_id=video_id,
            **kwargs
        )
    
    @staticmethod
    def template_not_found_error(template_id: str, **kwargs) -> ResourceNotFoundError:
        """Create template not found error"""
        return error_factory.resource_not_found_error(
            message=f"Template with ID '{template_id}' not found",
            resource_type="template",
            resource_id=template_id,
            **kwargs
        )
    
    @staticmethod
    def voice_not_found_error(voice_id: str, **kwargs) -> ResourceNotFoundError:
        """Create voice not found error"""
        return error_factory.resource_not_found_error(
            message=f"Voice with ID '{voice_id}' not found",
            resource_type="voice",
            resource_id=voice_id,
            **kwargs
        )
    
    @staticmethod
    def user_quota_exceeded_error(user_id: str, quota_type: str, current_usage: int, limit: int, **kwargs) -> QuotaExceededError:
        """Create user quota exceeded error"""
        return error_factory.quota_exceeded_error(
            message=f"User quota exceeded for {quota_type}",
            quota_type=quota_type,
            current_usage=current_usage,
            limit=limit,
            details={"user_id": user_id},
            **kwargs
        )
    
    @staticmethod
    def content_violation_error(content_type: str, violation_reason: str, **kwargs) -> ContentModerationError:
        """Create content violation error"""
        return error_factory.content_moderation_error(
            message=f"Content violation detected in {content_type}",
            content_type=content_type,
            moderation_result="violation",
            flagged_content=violation_reason,
            **kwargs
        )
    
    @staticmethod
    def video_generation_timeout_error(video_id: str, timeout_duration: float, **kwargs) -> TimeoutError:
        """Create video generation timeout error"""
        return error_factory.timeout_error(
            message=f"Video generation timed out after {timeout_duration} seconds",
            timeout_duration=timeout_duration,
            operation="video_generation",
            details={"video_id": video_id},
            **kwargs
        )
    
    @staticmethod
    def voice_synthesis_failed_error(voice_id: str, language: str, text_length: int, **kwargs) -> VoiceSynthesisError:
        """Create voice synthesis failed error"""
        return error_factory.voice_synthesis_error(
            message="Voice synthesis failed",
            voice_id=voice_id,
            language=language,
            text_length=text_length,
            **kwargs
        )
    
    @staticmethod
    def template_processing_failed_error(template_id: str, template_type: str, processing_stage: str, **kwargs) -> TemplateProcessingError:
        """Create template processing failed error"""
        return error_factory.template_processing_error(
            message="Template processing failed",
            template_id=template_id,
            template_type=template_type,
            processing_stage=processing_stage,
            **kwargs
        )
    
    @staticmethod
    async def file_upload_failed_error(file_path: str, file_type: str, file_size: int, **kwargs) -> FileProcessingError:
        """Create file upload failed error"""
        return error_factory.file_processing_error(
            message="File upload failed",
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            operation="upload",
            **kwargs
        )
    
    @staticmethod
    def feature_deprecated_error(feature_name: str, deprecation_date: datetime, replacement_feature: str, **kwargs) -> DeprecatedFeatureError:
        """Create feature deprecated error"""
        return error_factory.deprecated_feature_error(
            message=f"Feature '{feature_name}' has been deprecated",
            feature_name=feature_name,
            deprecation_date=deprecation_date,
            replacement_feature=replacement_feature,
            **kwargs
        )


# Global error factory instance
error_factory = ErrorFactory()


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        
    """__init__ function."""
self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN for {self.service_name}",
                    service_name=self.service_name,
                    failure_count=self.failure_count
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> Any:
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> Any:
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout


class RetryHandler:
    """Retry mechanism with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        
    """__init__ function."""
self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    async def execute(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    raise RetryExhaustedError(
                        f"Max retries ({self.max_retries}) exceeded for {operation_name or 'operation'}",
                        max_retries=self.max_retries,
                        attempts_made=attempt + 1
                    )
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {operation_name or 'operation'}, retrying in {delay}s")
                await asyncio.sleep(delay)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)


def handle_errors(
    category: Optional[ErrorCategory] = None,
    operation: Optional[str] = None,
    log_errors: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 3,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """Decorator for handling errors with logging and retry logic"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            request_id = str(uuid.uuid4())
            user_id = None
            
            # Extract user_id from kwargs if available
            for arg in args:
                if hasattr(arg, 'user_id'):
                    user_id = arg.user_id
                    break
            
            # Log operation start
            if log_errors:
                ErrorLogger.log_user_action(
                    action=f"start_{operation or func.__name__}",
                    user_id=user_id,
                    request_id=request_id,
                    success=True
                )
            
            try:
                # Execute with circuit breaker if provided
                if circuit_breaker:
                    result = circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                
                # Log successful operation
                if log_errors:
                    ErrorLogger.log_user_action(
                        action=f"complete_{operation or func.__name__}",
                        user_id=user_id,
                        request_id=request_id,
                        success=True
                    )
                
                return result
            
            except Exception as e:
                # Log error with context
                if log_errors:
                    ErrorLogger.log_error(
                        error=e,
                        context={
                            'operation': operation or func.__name__,
                            'category': category.value if category else 'unknown',
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        },
                        user_id=user_id,
                        request_id=request_id,
                        operation=operation or func.__name__
                    )
                
                # Log failed operation
                if log_errors:
                    ErrorLogger.log_user_action(
                        action=f"fail_{operation or func.__name__}",
                        user_id=user_id,
                        request_id=request_id,
                        success=False,
                        details={'error': str(e)}
                    )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            request_id = str(uuid.uuid4())
            user_id = None
            
            # Extract user_id from kwargs if available
            for arg in args:
                if hasattr(arg, 'user_id'):
                    user_id = arg.user_id
                    break
            
            # Log operation start
            if log_errors:
                ErrorLogger.log_user_action(
                    action=f"start_{operation or func.__name__}",
                    user_id=user_id,
                    request_id=request_id,
                    success=True
                )
            
            try:
                # Execute with circuit breaker if provided
                if circuit_breaker:
                    result = circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Log successful operation
                if log_errors:
                    ErrorLogger.log_user_action(
                        action=f"complete_{operation or func.__name__}",
                        user_id=user_id,
                        request_id=request_id,
                        success=True
                    )
                
                return result
            
            except Exception as e:
                # Log error with context
                if log_errors:
                    ErrorLogger.log_error(
                        error=e,
                        context={
                            'operation': operation or func.__name__,
                            'category': category.value if category else 'unknown',
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        },
                        user_id=user_id,
                        request_id=request_id,
                        operation=operation or func.__name__
                    )
                
                # Log failed operation
                if log_errors:
                    ErrorLogger.log_user_action(
                        action=f"fail_{operation or func.__name__}",
                        user_id=user_id,
                        request_id=request_id,
                        success=False,
                        details={'error': str(e)}
                    )
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Validation helper functions
def validate_input(
    field: str,
    value: Any,
    validator: Callable[[Any], bool],
    error_message: str,
    **kwargs
) -> None:
    """Validate input with custom validator"""
    if not validator(value):
        raise error_factory.validation_error(
            message=error_message,
            field=field,
            value=value,
            **kwargs
        )


def validate_required(
    field: str,
    value: Any,
    **kwargs
) -> None:
    """Validate that a field is required"""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise error_factory.validation_error(
            message=f"Field '{field}' is required",
            field=field,
            value=value,
            **kwargs
        )


def validate_length(
    field: str,
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs
) -> None:
    """Validate string length"""
    if not isinstance(value, str):
        raise error_factory.validation_error(
            message=f"Field '{field}' must be a string",
            field=field,
            value=value,
            **kwargs
        )
    
    if min_length is not None and len(value) < min_length:
        raise error_factory.validation_error(
            message=f"Field '{field}' must be at least {min_length} characters long",
            field=field,
            value=value,
            **kwargs
        )
    
    if max_length is not None and len(value) > max_length:
        raise error_factory.validation_error(
            message=f"Field '{field}' cannot exceed {max_length} characters",
            field=field,
            value=value,
            **kwargs
        )


def validate_enum(
    field: str,
    value: Any,
    enum_class: Type[Enum],
    **kwargs
) -> None:
    """Validate enum value"""
    if value not in enum_class:
        valid_values = [e.value for e in enum_class]
        raise error_factory.validation_error(
            message=f"Field '{field}' must be one of: {', '.join(map(str, valid_values))}",
            field=field,
            value=value,
            **kwargs
        )


def validate_range(
    field: str,
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    **kwargs
) -> None:
    """Validate numeric range"""
    if not isinstance(value, (int, float)):
        raise error_factory.validation_error(
            message=f"Field '{field}' must be a number",
            field=field,
            value=value,
            **kwargs
        )
    
    if min_value is not None and value < min_value:
        raise error_factory.validation_error(
            message=f"Field '{field}' must be at least {min_value}",
            field=field,
            value=value,
            **kwargs
        )
    
    if max_value is not None and value > max_value:
        raise error_factory.validation_error(
            message=f"Field '{field}' cannot exceed {max_value}",
            field=field,
            value=value,
            **kwargs
        )


# Exception handlers
async def heygen_exception_handler(request: Request, exc: HeyGenBaseError) -> JSONResponse:
    """Handle HeyGen AI exceptions with proper logging and user-friendly responses"""
    # Log the error with request context
    ErrorLogger.log_error(
        error=exc,
        context={
            'request_method': request.method,
            'request_url': str(request.url),
            'request_headers': dict(request.headers),
            'client_ip': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent')
        },
        user_id=getattr(request.state, 'user_id', None),
        request_id=getattr(request.state, 'request_id', None),
        operation=getattr(request.state, 'operation', None)
    )
    
    # Determine HTTP status code based on error category
    status_code_map = {
        ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,
        ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,
        ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,
        ErrorCategory.RESOURCE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ErrorCategory.RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorCategory.TIMEOUT: status.HTTP_408_REQUEST_TIMEOUT,
        ErrorCategory.DATABASE: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.EXTERNAL_SERVICE: status.HTTP_502_BAD_GATEWAY,
        ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.VIDEO_PROCESSING: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.CIRCUIT_BREAKER: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.RETRY_EXHAUSTED: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.CONCURRENCY: status.HTTP_409_CONFLICT,
        ErrorCategory.RESOURCE_EXHAUSTION: status.HTTP_503_SERVICE_UNAVAILABLE
    }
    
    http_status = status_code_map.get(exc.category, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Prepare response headers
    headers = {
        'X-Error-ID': exc.error_id,
        'X-Error-Category': exc.category.value,
        'Content-Type': 'application/json'
    }
    
    if exc.retry_after:
        headers['Retry-After'] = str(exc.retry_after)
    
    # Prepare response body
    response_body = {
        'error': {
            'id': exc.error_id,
            'code': exc.error_code,
            'message': exc.user_friendly_message,
            'category': exc.category.value,
            'severity': exc.severity.value,
            'timestamp': exc.timestamp.isoformat(),
            'details': exc.details,
            'retry_after': exc.retry_after
        }
    }
    
    return JSONResponse(
        status_code=http_status,
        content=response_body,
        headers=headers
    )


async def pydantic_validation_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with user-friendly messages"""
    # Convert Pydantic errors to user-friendly format
    validation_errors = []
    for error in exc.errors():
        field = '.'.join(str(loc) for loc in error['loc'])
        message = error['msg']
        
        # Generate user-friendly message
        if 'required' in message.lower():
            user_message = f"The field '{field}' is required."
        elif 'type' in message.lower():
            user_message = f"The field '{field}' has an invalid type."
        elif 'length' in message.lower():
            user_message = f"The field '{field}' has an invalid length."
        else:
            user_message = f"The field '{field}' is invalid: {message}"
        
        validation_errors.append({
            'field': field,
            'message': user_message,
            'type': error['type']
        })
    
    # Log the validation error
    ErrorLogger.log_error(
        error=ValidationError(
            message="Pydantic validation failed",
            validation_errors=[err['message'] for err in validation_errors]
        ),
        context={
            'request_method': request.method,
            'request_url': str(request.url),
            'validation_errors': validation_errors
        },
        user_id=getattr(request.state, 'user_id', None),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    response_body = {
        'error': {
            'id': str(uuid.uuid4()),
            'code': 'VALIDATION_ERROR',
            'message': 'Please check your input and try again.',
            'category': 'validation',
            'severity': 'low',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'validation_errors': validation_errors
            }
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response_body,
        headers={'X-Error-ID': response_body['error']['id']}
    )


async async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with user-friendly messages"""
    # Generate user-friendly message based on status code
    user_message_map = {
        400: "The request is invalid. Please check your input and try again.",
        401: "Please log in to continue.",
        403: "You don't have permission to perform this action.",
        404: "The requested resource was not found.",
        405: "This operation is not allowed.",
        408: "The request took too long to process. Please try again.",
        429: "Too many requests. Please try again later.",
        500: "Something went wrong on our end. Please try again later.",
        502: "We're experiencing technical difficulties. Please try again later.",
        503: "The service is temporarily unavailable. Please try again later."
    }
    
    user_message = user_message_map.get(exc.status_code, "An error occurred. Please try again.")
    
    # Log the HTTP exception
    ErrorLogger.log_error(
        error=exc,
        context={
            'request_method': request.method,
            'request_url': str(request.url),
            'status_code': exc.status_code
        },
        user_id=getattr(request.state, 'user_id', None),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    response_body = {
        'error': {
            'id': str(uuid.uuid4()),
            'code': f'HTTP_{exc.status_code}',
            'message': user_message,
            'category': 'http_exception',
            'severity': 'medium',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'status_code': exc.status_code,
                'detail': exc.detail
            }
        }
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_body,
        headers={'X-Error-ID': response_body['error']['id']}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with proper logging"""
    # Log the unexpected error
    ErrorLogger.log_error(
        error=exc,
        context={
            'request_method': request.method,
            'request_url': str(request.url),
            'exception_type': type(exc).__name__,
            'stack_trace': traceback.format_exc()
        },
        user_id=getattr(request.state, 'user_id', None),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    response_body = {
        'error': {
            'id': str(uuid.uuid4()),
            'code': 'INTERNAL_SERVER_ERROR',
            'message': 'Something went wrong on our end. Please try again later.',
            'category': 'system',
            'severity': 'high',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'exception_type': type(exc).__name__
            }
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_body,
        headers={'X-Error-ID': response_body['error']['id']}
    )


# Named exports
__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "HeyGenBaseError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "DatabaseError",
    "ResourceNotFoundError",
    "VideoProcessingError",
    "RateLimitError",
    "ExternalServiceError",
    "TimeoutError",
    "CircuitBreakerError",
    "RetryExhaustedError",
    "ConcurrencyError",
    "ResourceExhaustionError",
    "ErrorFactory",
    "error_factory",
    "CircuitBreaker",
    "RetryHandler",
    "handle_errors",
    "validate_input",
    "validate_required",
    "validate_length",
    "validate_enum",
    "validate_range",
    "heygen_exception_handler",
    "pydantic_validation_handler",
    "http_exception_handler",
    "general_exception_handler",
    "ErrorLogger",
    "UserFriendlyMessageGenerator"
] 