"""
Enhanced Logging Configuration for Video-OpusClip

Provides structured logging, user-friendly error messages, and comprehensive error tracking.
"""

import logging
import logging.config
import structlog
import sys
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import uuid

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = True,
    enable_request_tracking: bool = True
) -> None:
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Enable console logging
        enable_json: Enable JSON structured logging
        enable_request_tracking: Enable request ID tracking
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if enable_request_tracking:
        processors.insert(0, add_request_id)
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Add file handler if specified
    if log_file:
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        logging_config["loggers"][""]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

def add_request_id(logger, method_name, event_dict):
    """Add request ID to log entries if available."""
    request_id = getattr(logger, 'request_id', None)
    if request_id:
        event_dict['request_id'] = request_id
    return event_dict

# =============================================================================
# ENHANCED LOGGER CLASS
# =============================================================================

class EnhancedLogger:
    """Enhanced logger with user-friendly error messages and structured logging."""
    
    def __init__(self, name: str, request_id: Optional[str] = None):
        self.logger = structlog.get_logger(name)
        self.request_id = request_id or str(uuid.uuid4())
        self.logger.request_id = self.request_id
    
    def set_request_id(self, request_id: str) -> None:
        """Set request ID for tracking."""
        self.request_id = request_id
        self.logger.request_id = request_id
    
    def _format_user_message(self, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Format user-friendly error message."""
        if not details:
            return message
        
        # Add helpful context
        context_parts = []
        
        if 'field' in details:
            context_parts.append(f"Field: {details['field']}")
        
        if 'value' in details and details['value'] is not None:
            value_str = str(details['value'])
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            context_parts.append(f"Value: {value_str}")
        
        if 'operation' in details:
            context_parts.append(f"Operation: {details['operation']}")
        
        if 'service' in details:
            context_parts.append(f"Service: {details['service']}")
        
        if 'resource' in details:
            context_parts.append(f"Resource: {details['resource']}")
        
        if context_parts:
            return f"{message} ({', '.join(context_parts)})"
        
        return message
    
    def _format_technical_message(self, message: str, error: Optional[Exception] = None, 
                                 details: Optional[Dict[str, Any]] = None) -> str:
        """Format technical error message for developers."""
        parts = [message]
        
        if error:
            parts.append(f"Exception: {type(error).__name__}: {str(error)}")
        
        if details:
            for key, value in details.items():
                if value is not None:
                    parts.append(f"{key}: {value}")
        
        return " | ".join(parts)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, request_id=self.request_id, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, request_id=self.request_id, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, 
              details: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log error message with user-friendly and technical details."""
        # User-friendly message
        user_message = self._format_user_message(message, details)
        
        # Technical message for developers
        tech_message = self._format_technical_message(message, error, details)
        
        # Log both messages
        self.logger.error(
            user_message,
            technical_details=tech_message,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            stack_trace=traceback.format_exc() if error else None,
            request_id=self.request_id,
            **kwargs
        )
    
    def critical(self, message: str, error: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log critical error message."""
        # User-friendly message
        user_message = self._format_user_message(message, details)
        
        # Technical message for developers
        tech_message = self._format_technical_message(message, error, details)
        
        # Log both messages
        self.logger.critical(
            user_message,
            technical_details=tech_message,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            stack_trace=traceback.format_exc() if error else None,
            request_id=self.request_id,
            **kwargs
        )
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, request_id=self.request_id, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with full stack trace."""
        self.logger.exception(message, request_id=self.request_id, **kwargs)

# =============================================================================
# ERROR MESSAGE TEMPLATES
# =============================================================================

class ErrorMessages:
    """User-friendly error message templates."""
    
    # Validation Errors
    VALIDATION_ERRORS = {
        "youtube_url_required": "Please provide a valid YouTube URL",
        "youtube_url_invalid": "The YouTube URL format is not valid. Please check the URL and try again",
        "youtube_url_too_long": "The YouTube URL is too long. Please use a shorter URL",
        "youtube_url_malicious": "The URL contains potentially harmful content and cannot be processed",
        
        "language_required": "Please specify a language for the video processing",
        "language_invalid": "The language code is not supported. Please use a valid language code (e.g., 'en', 'es', 'fr')",
        
        "clip_length_invalid": "The clip length must be between {min_length} and {max_length} seconds",
        "clip_length_negative": "Clip length cannot be negative",
        "clip_length_zero": "Clip length cannot be zero",
        "clip_length_too_long": "The requested clip length exceeds the maximum allowed duration",
        
        "viral_score_invalid": "Viral score must be between 0.0 and 1.0",
        
        "caption_required": "Please provide a caption for the video",
        "caption_empty": "Caption cannot be empty",
        "caption_too_long": "Caption is too long. Maximum length is {max_length} characters",
        
        "variant_id_required": "Variant ID is required",
        "variant_id_invalid": "Variant ID contains invalid characters",
        
        "batch_size_invalid": "Batch size must be between {min_size} and {max_size}",
        "batch_empty": "Batch cannot be empty",
        "batch_too_large": "Batch size exceeds the maximum allowed limit",
        
        # Image generation specific validation errors
        "prompt_required": "Please provide a text prompt for image generation",
        "prompt_empty": "The prompt cannot be empty. Please enter some text to describe the image you want to generate",
        "prompt_invalid_type": "The prompt must be text. Please enter a valid description",
        "prompt_too_long": "The prompt is too long. Please use {max_length} characters or fewer",
        
        "guidance_scale_invalid_type": "Guidance scale must be a number",
        "guidance_scale_out_of_range": "Guidance scale must be between {min_value} and {max_value}",
        
        "inference_steps_invalid_type": "Inference steps must be a whole number",
        "inference_steps_out_of_range": "Inference steps must be between {min_value} and {max_value}",
    }
    
    # Processing Errors
    PROCESSING_ERRORS = {
        "video_processing_failed": "Video processing failed. Please try again later",
        "video_processing_timeout": "Video processing took too long. Please try with a shorter video",
        "video_encoding_failed": "Failed to encode the video. Please check the video format",
        "audio_extraction_failed": "Failed to extract audio from the video",
        "frame_extraction_failed": "Failed to extract frames from the video",
        
        "langchain_processing_failed": "AI analysis failed. Please try again later",
        "langchain_timeout": "AI analysis took too long. Please try again",
        "langchain_api_error": "AI service is temporarily unavailable. Please try again later",
        
        "viral_analysis_failed": "Viral analysis failed. Please try again later",
        "batch_processing_failed": "Batch processing failed. Some videos may not have been processed",
        
        # Image generation specific processing errors
        "image_generation_failed": "Image generation failed. Please try again with different parameters",
        "pipeline_not_available": "Image generation system is not available. Please check system configuration",
        "pipeline_invalid_result": "Image generation returned an invalid result. Please try again",
        "no_images_generated": "No images were generated. Please try again with a different prompt",
        "generated_image_none": "Image generation failed to produce a valid image. Please try again",
        "generated_image_invalid_dimensions": "Generated image has invalid dimensions. Please try again",
        "generated_image_invalid_type": "Generated image is not in the correct format. Please try again",
    }
    
    # Resource Errors
    RESOURCE_ERRORS = {
        "insufficient_memory": "System memory is insufficient. Please try again later",
        "gpu_not_available": "GPU processing is not available. Using CPU instead",
        "gpu_memory_full": "GPU memory is full. Please try again later or use CPU processing",
        "gpu_memory_critical": "GPU memory usage is critical. Please try again later or reduce parameters",
        "gpu_memory_insufficient": "GPU memory is insufficient. Try reducing image size or batch size",
        "disk_space_full": "System storage is full. Please try again later",
        "rate_limit_exceeded": "Too many requests. Please wait a moment and try again",
        "timeout_error": "Request timed out. Please try again",
        "cpu_overloaded": "System is currently busy. Please try again later",
    }
    
    # External Service Errors
    EXTERNAL_SERVICE_ERRORS = {
        "youtube_api_error": "YouTube service is temporarily unavailable. Please try again later",
        "youtube_video_not_found": "The YouTube video could not be found. Please check the URL",
        "youtube_video_private": "The YouTube video is private and cannot be accessed",
        "youtube_video_restricted": "The YouTube video is restricted and cannot be processed",
        
        "langchain_api_error": "AI service is temporarily unavailable. Please try again later",
        "langchain_rate_limit": "AI service is busy. Please wait a moment and try again",
        
        "redis_connection_error": "Cache service is temporarily unavailable",
        "database_error": "Database service is temporarily unavailable",
        "file_storage_error": "File storage service is temporarily unavailable",
    }
    
    # Security Errors
    SECURITY_ERRORS = {
        "unauthorized_access": "Access denied. Please check your credentials",
        "invalid_token": "Invalid authentication token. Please log in again",
        "rate_limit_violation": "Too many requests from your IP. Please wait before trying again",
        "malicious_input_detected": "Potentially harmful input detected. Request blocked for security",
        "injection_attempt": "Invalid input detected. Please check your request",
        "excessive_requests": "Too many requests. Please wait before trying again",
    }
    
    # Configuration Errors
    CONFIGURATION_ERRORS = {
        "missing_config": "System configuration is missing. Please contact support",
        "invalid_config": "System configuration is invalid. Please contact support",
        "environment_error": "System environment is not properly configured. Please contact support",
        "model_config_invalid": "AI model configuration is invalid. Please contact support",
        "api_key_missing": "API key is missing. Please check your configuration",
        "environment_variable_missing": "Required environment variable is missing. Please contact support",
    }
    
    # System Errors
    SYSTEM_ERRORS = {
        "system_crash": "System error occurred. Please try again later",
        "database_connection_lost": "Database connection lost. Please try again",
        "redis_connection_lost": "Cache connection lost. Please try again",
        "gpu_memory_exhausted": "GPU memory exhausted. Please try again later",
        "disk_space_critical": "System storage is critically low. Please try again later",
        "model_loading_failed": "AI model failed to load. Please try again later",
        "pipeline_initialization_failed": "Processing pipeline failed to initialize. Please try again later",
        "critical_service_unavailable": "Critical service is unavailable. Please try again later",
        "unexpected_error": "An unexpected error occurred. Please try again later",
    }
    
    @classmethod
    def get_user_message(cls, error_code: str, **kwargs) -> str:
        """Get user-friendly error message for given error code."""
        # Check all error categories
        for error_dict in [
            cls.VALIDATION_ERRORS,
            cls.PROCESSING_ERRORS,
            cls.RESOURCE_ERRORS,
            cls.EXTERNAL_SERVICE_ERRORS,
            cls.SECURITY_ERRORS,
            cls.CONFIGURATION_ERRORS,
            cls.SYSTEM_ERRORS
        ]:
            if error_code in error_dict:
                message = error_dict[error_code]
                try:
                    return message.format(**kwargs)
                except KeyError:
                    return message
        
        # Default message
        return "An unexpected error occurred. Please try again later"
    
    @classmethod
    def get_suggestion(cls, error_code: str) -> Optional[str]:
        """Get helpful suggestion for error resolution."""
        suggestions = {
            "youtube_url_invalid": "Make sure the URL starts with 'https://www.youtube.com/watch?v=' or 'https://youtu.be/'",
            "youtube_url_malicious": "Please use a different YouTube URL",
            "language_invalid": "Supported languages: en, es, fr, de, it, pt, ru, ja, ko, zh",
            "clip_length_invalid": "Try a clip length between 10 and 600 seconds",
            "caption_too_long": "Try a shorter caption (maximum 1000 characters)",
            "batch_too_large": "Try processing fewer videos at once (maximum 100)",
            "rate_limit_exceeded": "Wait 1-2 minutes before trying again",
            "gpu_not_available": "Processing will continue with CPU (may be slower)",
            "youtube_video_not_found": "Check if the video URL is correct and the video is public",
            "youtube_video_private": "Only public YouTube videos can be processed",
            "malicious_input_detected": "Check your input for any suspicious content",
                    "insufficient_memory": "Try processing a shorter video or wait for system resources to free up",
        "gpu_memory_critical": "Try reducing the number of inference steps or guidance scale",
        "gpu_memory_insufficient": "Try reducing image size, batch size, or number of inference steps",
        "pipeline_not_available": "Contact support if this issue persists",
        "image_generation_failed": "Try using a different prompt or adjusting the parameters",
        "unexpected_error": "If this error persists, please contact support with the request ID",
    }
        
        return suggestions.get(error_code)

# =============================================================================
# ERROR LOGGING UTILITIES
# =============================================================================

def log_error_with_context(
    logger: EnhancedLogger,
    error: Exception,
    context: Dict[str, Any],
    user_message: Optional[str] = None,
    error_code: Optional[str] = None
) -> None:
    """Log error with comprehensive context and user-friendly message."""
    
    # Get error code from exception if available
    if not error_code and hasattr(error, 'error_code'):
        error_code = error.error_code.name.lower()
    
    # Get user-friendly message
    if not user_message:
        user_message = ErrorMessages.get_user_message(error_code or "unknown_error")
    
    # Get suggestion
    suggestion = ErrorMessages.get_suggestion(error_code) if error_code else None
    
    # Log error with full context
    logger.error(
        user_message,
        error=error,
        details={
            "error_code": error_code,
            "suggestion": suggestion,
            **context
        }
    )

def log_validation_error(
    logger: EnhancedLogger,
    field: str,
    value: Any,
    error_code: str,
    message: str
) -> None:
    """Log validation error with user-friendly message."""
    user_message = ErrorMessages.get_user_message(error_code, field=field)
    suggestion = ErrorMessages.get_suggestion(error_code)
    
    logger.warning(
        user_message,
        details={
            "field": field,
            "value": value,
            "error_code": error_code,
            "suggestion": suggestion,
            "technical_message": message
        }
    )

def log_processing_error(
    logger: EnhancedLogger,
    operation: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log processing error with context."""
    error_code = getattr(error, 'error_code', None)
    if error_code:
        error_code = error_code.name.lower()
    
    user_message = ErrorMessages.get_user_message(error_code or "processing_failed")
    suggestion = ErrorMessages.get_suggestion(error_code) if error_code else None
    
    logger.error(
        user_message,
        error=error,
        details={
            "operation": operation,
            "error_code": error_code,
            "suggestion": suggestion,
            **(context or {})
        }
    )

def log_resource_error(
    logger: EnhancedLogger,
    resource: str,
    available: Optional[Any] = None,
    required: Optional[Any] = None,
    error: Optional[Exception] = None
) -> None:
    """Log resource error with context."""
    error_code = "insufficient_resource"
    user_message = ErrorMessages.get_user_message(error_code, resource=resource)
    suggestion = ErrorMessages.get_suggestion(error_code)
    
    logger.warning(
        user_message,
        error=error,
        details={
            "resource": resource,
            "available": available,
            "required": required,
            "error_code": error_code,
            "suggestion": suggestion
        }
    )

def log_security_error(
    logger: EnhancedLogger,
    threat_type: str,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None
) -> None:
    """Log security error with context."""
    error_code = "security_violation"
    user_message = ErrorMessages.get_user_message(error_code)
    suggestion = ErrorMessages.get_suggestion(error_code)
    
    logger.warning(
        user_message,
        error=error,
        details={
            "threat_type": threat_type,
            "error_code": error_code,
            "suggestion": suggestion,
            **(details or {})
        }
    )

# =============================================================================
# REQUEST TRACKING
# =============================================================================

class RequestTracker:
    """Track request context for logging."""
    
    def __init__(self):
        self.request_id = None
        self.user_id = None
        self.session_id = None
        self.start_time = None
    
    def start_request(self, request_id: str, user_id: Optional[str] = None, 
                     session_id: Optional[str] = None) -> None:
        """Start tracking a new request."""
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = datetime.utcnow()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current request context."""
        context = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id
        }
        
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            context["duration"] = duration
        
        return context
    
    def clear(self) -> None:
        """Clear request context."""
        self.request_id = None
        self.user_id = None
        self.session_id = None
        self.start_time = None

# Global request tracker
request_tracker = RequestTracker()

# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

def create_logging_middleware():
    """Create FastAPI middleware for request logging."""
    from fastapi import Request
    import time
    
    async def logging_middleware(request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start tracking
        request_tracker.start_request(request_id)
        
        # Create logger for this request
        logger = EnhancedLogger("api", request_id)
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log request completion
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration=duration
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log request error
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                error=e,
                duration=duration
            )
            raise
        
        finally:
            # Clear request context
            request_tracker.clear()
    
    return logging_middleware

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/video_opusclip.log",
    enable_console: bool = True,
    enable_json: bool = True
) -> None:
    """Initialize logging for the application."""
    
    # Setup logging configuration
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_json=enable_json
    )
    
    # Create main logger
    main_logger = EnhancedLogger("video_opusclip")
    
    # Log initialization
    main_logger.info(
        "Logging system initialized",
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_json=enable_json
    )

# Initialize logging when module is imported
if __name__ != "__main__":
    initialize_logging() 