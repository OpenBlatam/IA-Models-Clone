"""
Custom Exceptions for OpusClip Improved
======================================

Comprehensive exception handling with detailed error information.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class OpusClipException(Exception):
    """Base exception for OpusClip operations"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "OPUS_CLIP_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)


class VideoProcessingError(OpusClipException):
    """Error during video processing"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        processing_stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.video_id = video_id
        self.processing_stage = processing_stage
        error_details = details or {}
        if video_id:
            error_details["video_id"] = video_id
        if processing_stage:
            error_details["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code="VIDEO_PROCESSING_ERROR",
            details=error_details
        )


class VideoAnalysisError(OpusClipException):
    """Error during video analysis"""
    
    def __init__(
        self,
        message: str,
        analysis_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.analysis_id = analysis_id
        self.analysis_type = analysis_type
        error_details = details or {}
        if analysis_id:
            error_details["analysis_id"] = analysis_id
        if analysis_type:
            error_details["analysis_type"] = analysis_type
        
        super().__init__(
            message=message,
            error_code="VIDEO_ANALYSIS_ERROR",
            details=error_details
        )


class ClipGenerationError(OpusClipException):
    """Error during clip generation"""
    
    def __init__(
        self,
        message: str,
        generation_id: Optional[str] = None,
        clip_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.generation_id = generation_id
        self.clip_type = clip_type
        error_details = details or {}
        if generation_id:
            error_details["generation_id"] = generation_id
        if clip_type:
            error_details["clip_type"] = clip_type
        
        super().__init__(
            message=message,
            error_code="CLIP_GENERATION_ERROR",
            details=error_details
        )


class AIProviderError(OpusClipException):
    """Error with AI provider"""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.model = model
        error_details = details or {}
        if provider:
            error_details["provider"] = provider
        if model:
            error_details["model"] = model
        
        super().__init__(
            message=message,
            error_code="AI_PROVIDER_ERROR",
            details=error_details
        )


class ValidationError(OpusClipException):
    """Input validation error"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.field = field
        self.value = value
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class FileError(OpusClipException):
    """File operation error"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.file_path = file_path
        self.operation = operation
        error_details = details or {}
        if file_path:
            error_details["file_path"] = file_path
        if operation:
            error_details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="FILE_ERROR",
            details=error_details
        )


class DatabaseError(OpusClipException):
    """Database operation error"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.table = table
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if table:
            error_details["table"] = table
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=error_details
        )


class RateLimitError(OpusClipException):
    """Rate limit exceeded error"""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        error_details = details or {}
        if limit:
            error_details["limit"] = limit
        if window:
            error_details["window"] = window
        if retry_after:
            error_details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=error_details
        )


class AuthenticationError(OpusClipException):
    """Authentication error"""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        error_details = details or {}
        if user_id:
            error_details["user_id"] = user_id
        
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=error_details
        )


class AuthorizationError(OpusClipException):
    """Authorization error"""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.resource = resource
        self.action = action
        error_details = details or {}
        if user_id:
            error_details["user_id"] = user_id
        if resource:
            error_details["resource"] = resource
        if action:
            error_details["action"] = action
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=error_details
        )


class ConfigurationError(OpusClipException):
    """Configuration error"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.config_key = config_key
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details
        )


class ExternalServiceError(OpusClipException):
    """External service error"""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.service = service
        self.endpoint = endpoint
        self.status_code = status_code
        error_details = details or {}
        if service:
            error_details["service"] = service
        if endpoint:
            error_details["endpoint"] = endpoint
        if status_code:
            error_details["status_code"] = status_code
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=error_details
        )


class TimeoutError(OpusClipException):
    """Operation timeout error"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_duration: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.timeout_duration = timeout_duration
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if timeout_duration:
            error_details["timeout_duration"] = timeout_duration
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=error_details
        )


class ResourceNotFoundError(OpusClipException):
    """Resource not found error"""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=error_details
        )


class QuotaExceededError(OpusClipException):
    """Quota exceeded error"""
    
    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        quota_limit: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        error_details = details or {}
        if quota_type:
            error_details["quota_type"] = quota_type
        if current_usage:
            error_details["current_usage"] = current_usage
        if quota_limit:
            error_details["quota_limit"] = quota_limit
        
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            details=error_details
        )


class BatchProcessingError(OpusClipException):
    """Batch processing error"""
    
    def __init__(
        self,
        message: str,
        batch_id: Optional[str] = None,
        failed_items: Optional[int] = None,
        total_items: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.batch_id = batch_id
        self.failed_items = failed_items
        self.total_items = total_items
        error_details = details or {}
        if batch_id:
            error_details["batch_id"] = batch_id
        if failed_items:
            error_details["failed_items"] = failed_items
        if total_items:
            error_details["total_items"] = total_items
        
        super().__init__(
            message=message,
            error_code="BATCH_PROCESSING_ERROR",
            details=error_details
        )


class ExportError(OpusClipException):
    """Export operation error"""
    
    def __init__(
        self,
        message: str,
        export_id: Optional[str] = None,
        export_format: Optional[str] = None,
        target_platform: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.export_id = export_id
        self.export_format = export_format
        self.target_platform = target_platform
        error_details = details or {}
        if export_id:
            error_details["export_id"] = export_id
        if export_format:
            error_details["export_format"] = export_format
        if target_platform:
            error_details["target_platform"] = target_platform
        
        super().__init__(
            message=message,
            error_code="EXPORT_ERROR",
            details=error_details
        )


# Error factory functions for common error scenarios
def create_video_processing_error(
    stage: str,
    video_id: str,
    original_error: Exception
) -> VideoProcessingError:
    """Create a video processing error with context"""
    return VideoProcessingError(
        message=f"Video processing failed at {stage}: {str(original_error)}",
        video_id=video_id,
        processing_stage=stage,
        details={"original_error": str(original_error)}
    )


def create_ai_provider_error(
    provider: str,
    model: str,
    original_error: Exception
) -> AIProviderError:
    """Create an AI provider error with context"""
    return AIProviderError(
        message=f"AI provider {provider} failed with model {model}: {str(original_error)}",
        provider=provider,
        model=model,
        details={"original_error": str(original_error)}
    )


def create_validation_error(
    field: str,
    value: Any,
    expected_type: str
) -> ValidationError:
    """Create a validation error with context"""
    return ValidationError(
        message=f"Invalid value for field '{field}': expected {expected_type}, got {type(value).__name__}",
        field=field,
        value=value,
        details={"expected_type": expected_type}
    )


def create_file_error(
    file_path: str,
    operation: str,
    original_error: Exception
) -> FileError:
    """Create a file error with context"""
    return FileError(
        message=f"File operation '{operation}' failed for '{file_path}': {str(original_error)}",
        file_path=file_path,
        operation=operation,
        details={"original_error": str(original_error)}
    )


def create_rate_limit_error(
    limit: int,
    window: int,
    retry_after: int
) -> RateLimitError:
    """Create a rate limit error with context"""
    return RateLimitError(
        message=f"Rate limit exceeded: {limit} requests per {window} seconds",
        limit=limit,
        window=window,
        retry_after=retry_after
    )


def create_resource_not_found_error(
    resource_type: str,
    resource_id: str
) -> ResourceNotFoundError:
    """Create a resource not found error with context"""
    return ResourceNotFoundError(
        message=f"{resource_type} with ID '{resource_id}' not found",
        resource_type=resource_type,
        resource_id=resource_id
    )


def create_quota_exceeded_error(
    quota_type: str,
    current_usage: int,
    quota_limit: int
) -> QuotaExceededError:
    """Create a quota exceeded error with context"""
    return QuotaExceededError(
        message=f"{quota_type} quota exceeded: {current_usage}/{quota_limit}",
        quota_type=quota_type,
        current_usage=current_usage,
        quota_limit=quota_limit
    )






























