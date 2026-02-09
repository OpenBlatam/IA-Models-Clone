from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import functools
import inspect
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
import gradio as gr
from pydantic import BaseModel, ValidationError as PydanticValidationError
import numpy as np
from PIL import Image
from .core import (
from .models import VideoRequest, VideoResponse
from .utils.validation import validate_file_type, validate_file_size
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Gradio Error Handling and Input Validation System

Comprehensive error handling and input validation for Gradio applications
in the AI Video system. Provides robust error management, user-friendly
error messages, and input validation with proper error categorization.
"""

    Any, Callable, Dict, List, Optional, Tuple, Union, 
    TypeVar, ParamSpec, Awaitable
)

# Local imports
    AIVideoError, ValidationError, ConfigurationError, WorkflowError,
    main_logger, performance_logger, security_logger
)

# Type variables for decorators
T = TypeVar('T')
P = ParamSpec('P')

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for proper handling."""
    INPUT_VALIDATION = "input_validation"
    FILE_PROCESSING = "file_processing"
    MODEL_LOADING = "model_loading"
    GPU_MEMORY = "gpu_memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class GradioErrorInfo(BaseModel):
    """Structured error information for Gradio applications."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    user_message: str
    technical_message: str
    error_code: Optional[str] = None
    suggestions: List[str] = []
    retry_allowed: bool = True
    max_retries: int = 3


class InputValidationRule(BaseModel):
    """Validation rule for input parameters."""
    field_name: str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    file_types: Optional[List[str]] = None
    max_file_size: Optional[int] = None  # in bytes
    custom_validator: Optional[Callable] = None


class GradioErrorHandler:
    """
    Comprehensive error handler for Gradio applications.
    
    Provides structured error handling, user-friendly messages,
    and proper error categorization for the AI Video system.
    """
    
    def __init__(self) -> Any:
        self.error_history: List[GradioErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.max_history_size = 1000
        
        # Error message templates
        self.error_templates = {
            ErrorCategory.INPUT_VALIDATION: {
                "title": "Invalid Input",
                "description": "Please check your input and try again.",
                "icon": "⚠️"
            },
            ErrorCategory.FILE_PROCESSING: {
                "title": "File Processing Error",
                "description": "There was an issue processing your file.",
                "icon": "📁"
            },
            ErrorCategory.MODEL_LOADING: {
                "title": "Model Loading Error",
                "description": "Unable to load the AI model.",
                "icon": "🤖"
            },
            ErrorCategory.GPU_MEMORY: {
                "title": "GPU Memory Error",
                "description": "Insufficient GPU memory for processing.",
                "icon": "💾"
            },
            ErrorCategory.NETWORK: {
                "title": "Network Error",
                "description": "Network connection issue.",
                "icon": "🌐"
            },
            ErrorCategory.CONFIGURATION: {
                "title": "Configuration Error",
                "description": "System configuration issue.",
                "icon": "⚙️"
            },
            ErrorCategory.PERMISSION: {
                "title": "Permission Error",
                "description": "Insufficient permissions.",
                "icon": "🔒"
            },
            ErrorCategory.TIMEOUT: {
                "title": "Timeout Error",
                "description": "Operation timed out.",
                "icon": "⏰"
            },
            ErrorCategory.UNKNOWN: {
                "title": "Unexpected Error",
                "description": "An unexpected error occurred.",
                "icon": "❌"
            }
        }
        
        # User-friendly error messages
        self.user_messages = {
            "text_too_short": "Please provide more detailed text (minimum {min_length} characters).",
            "text_too_long": "Text is too long (maximum {max_length} characters).",
            "invalid_quality": "Please select a valid quality setting: {allowed_values}.",
            "invalid_duration": "Duration must be between {min_value} and {max_value} seconds.",
            "file_too_large": "File is too large (maximum {max_size} MB).",
            "invalid_file_type": "File type not supported. Allowed types: {allowed_types}.",
            "gpu_memory_full": "GPU memory is full. Please try with lower quality or shorter duration.",
            "model_not_loaded": "AI model is not available. Please try again later.",
            "network_timeout": "Network request timed out. Please check your connection.",
            "permission_denied": "You don't have permission to perform this action.",
            "system_busy": "System is currently busy. Please try again in a moment.",
            "invalid_configuration": "System configuration error. Please contact support."
        }
    
    def create_error_info(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        user_message: str,
        technical_message: str,
        error_code: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> GradioErrorInfo:
        """Create structured error information."""
        error_id = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_history)}"
        
        return GradioErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            user_message=user_message,
            technical_message=technical_message,
            error_code=error_code,
            suggestions=suggestions or [],
            retry_allowed=self._can_retry(category, severity),
            max_retries=self._get_max_retries(category)
        )
    
    def _can_retry(self, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if retry is allowed for this error."""
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        non_retryable_categories = {
            ErrorCategory.PERMISSION,
            ErrorCategory.CONFIGURATION
        }
        
        return category not in non_retryable_categories
    
    def _get_max_retries(self, category: ErrorCategory) -> int:
        """Get maximum retry attempts for error category."""
        retry_config = {
            ErrorCategory.NETWORK: 3,
            ErrorCategory.TIMEOUT: 2,
            ErrorCategory.GPU_MEMORY: 1,
            ErrorCategory.MODEL_LOADING: 2,
            ErrorCategory.FILE_PROCESSING: 1,
            ErrorCategory.INPUT_VALIDATION: 0,  # No retry for validation errors
            ErrorCategory.PERMISSION: 0,
            ErrorCategory.CONFIGURATION: 0,
            ErrorCategory.UNKNOWN: 1
        }
        
        return retry_config.get(category, 1)
    
    def categorize_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error based on type and content."""
        error_str = str(error).lower()
        
        # GPU and memory errors
        if any(keyword in error_str for keyword in ["cuda", "gpu", "memory", "out of memory"]):
            return ErrorCategory.GPU_MEMORY, ErrorSeverity.HIGH
        
        # Network errors
        if any(keyword in error_str for keyword in ["connection", "timeout", "network", "http"]):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # File processing errors
        if any(keyword in error_str for keyword in ["file", "format", "corrupt", "invalid"]):
            return ErrorCategory.FILE_PROCESSING, ErrorSeverity.MEDIUM
        
        # Model loading errors
        if any(keyword in error_str for keyword in ["model", "load", "checkpoint", "weights"]):
            return ErrorCategory.MODEL_LOADING, ErrorSeverity.HIGH
        
        # Permission errors
        if any(keyword in error_str for keyword in ["permission", "access", "denied", "forbidden"]):
            return ErrorCategory.PERMISSION, ErrorSeverity.HIGH
        
        # Configuration errors
        if any(keyword in error_str for keyword in ["config", "setting", "parameter"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL
        
        # Timeout errors
        if "timeout" in error_str:
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        # Default to unknown
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def get_user_friendly_message(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get user-friendly error message."""
        context = context or {}
        
        # Try to get specific message based on error type
        if isinstance(error, ValidationError):
            return self._get_validation_message(error, context)
        elif isinstance(error, ConfigurationError):
            return self.user_messages["invalid_configuration"]
        elif isinstance(error, WorkflowError):
            return "Video generation workflow failed. Please try again."
        
        # Get message based on category
        error_str = str(error).lower()
        
        if category == ErrorCategory.GPU_MEMORY:
            return self.user_messages["gpu_memory_full"]
        elif category == ErrorCategory.MODEL_LOADING:
            return self.user_messages["model_not_loaded"]
        elif category == ErrorCategory.NETWORK:
            return self.user_messages["network_timeout"]
        elif category == ErrorCategory.PERMISSION:
            return self.user_messages["permission_denied"]
        elif category == ErrorCategory.TIMEOUT:
            return self.user_messages["network_timeout"]
        
        # Default message
        return "An unexpected error occurred. Please try again."
    
    def _get_validation_message(self, error: ValidationError, context: Dict[str, Any]) -> str:
        """Get specific validation error message."""
        error_str = str(error).lower()
        
        if "length" in error_str:
            if "minimum" in error_str:
                return self.user_messages["text_too_short"f"]"
                )
            elif "maximum" in error_str:
                return self.user_messages["text_too_long"f"]"
                )
        elif "quality" in error_str:
            return self.user_messages["invalid_quality"f"]")
            )
        elif "duration" in error_str:
            return self.user_messages["invalid_duration"f"]",
                max_value=context.get("max_value", 300)
            )
        elif "file" in error_str:
            if "size" in error_str:
                max_size_mb = context.get("max_size_mb", 100)
                return self.user_messages["file_too_large"f"]"
            elif "type" in error_str:
                allowed_types = context.get("allowed_types", ["mp4", "avi", "mov"])
                return self.user_messages["invalid_file_type"f"]"
                )
        
        return "Please check your input and try again."
    
    def get_suggestions(self, category: ErrorCategory, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get helpful suggestions for error resolution."""
        context = context or {}
        
        suggestions = {
            ErrorCategory.INPUT_VALIDATION: [
                "Check that all required fields are filled",
                "Ensure text length is within limits",
                "Verify file format and size"
            ],
            ErrorCategory.GPU_MEMORY: [
                "Try reducing video quality",
                "Use shorter video duration",
                "Close other applications using GPU",
                "Wait a moment and try again"
            ],
            ErrorCategory.MODEL_LOADING: [
                "Try again in a few moments",
                "Check your internet connection",
                "Contact support if problem persists"
            ],
            ErrorCategory.NETWORK: [
                "Check your internet connection",
                "Try again in a few moments",
                "Use a different network if available"
            ],
            ErrorCategory.FILE_PROCESSING: [
                "Check file format and size",
                "Try with a different file",
                "Ensure file is not corrupted"
            ],
            ErrorCategory.TIMEOUT: [
                "Try again with shorter content",
                "Check your internet connection",
                "Use lower quality settings"
            ],
            ErrorCategory.PERMISSION: [
                "Contact system administrator",
                "Check your account permissions",
                "Try logging in again"
            ],
            ErrorCategory.CONFIGURATION: [
                "Contact technical support",
                "Try refreshing the page",
                "Clear browser cache and try again"
            ]
        }
        
        return suggestions.get(category, ["Try again", "Contact support if problem persists"])
    
    def format_error_for_gradio(
        self,
        error: Exception,
        show_technical: bool = False
    ) -> Tuple[str, str, str]:
        """Format error for Gradio display."""
        category, severity = self.categorize_error(error)
        template = self.error_templates[category]
        
        # Create error info
        error_info = self.create_error_info(
            error=error,
            category=category,
            severity=severity,
            user_message=self.get_user_friendly_message(error, category),
            technical_message=str(error),
            suggestions=self.get_suggestions(category)
        )
        
        # Store error
        self._store_error(error_info)
        
        # Format for Gradio
        title = f"{template['icon']} {template['title']}"
        description = template['description']
        
        if show_technical:
            details = f"Technical Details:\n{error_info.technical_message}\n\nError ID: {error_info.error_id}"
        else:
            details = f"Error ID: {error_info.error_id}"
        
        if error_info.suggestions:
            details += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in error_info.suggestions)
        
        return title, description, details
    
    def _store_error(self, error_info: GradioErrorInfo) -> None:
        """Store error information."""
        self.error_history.append(error_info)
        
        # Limit history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        # Update error counts
        category_key = error_info.category.value
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Log error
        logger.error(
            f"Gradio Error [{error_info.error_id}]: {error_info.technical_message}",
            extra={
                "error_id": error_info.error_id,
                "category": error_info.category.value,
                "severity": error_info.severity.value,
                "user_message": error_info.user_message
            }
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": [
                {
                    "id": e.error_id,
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat(),
                    "user_message": e.user_message
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ]
        }


class GradioInputValidator:
    """
    Input validator for Gradio applications.
    
    Provides comprehensive input validation with proper error handling
    and user-friendly error messages.
    """
    
    def __init__(self, error_handler: GradioErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.validation_rules: Dict[str, InputValidationRule] = {}
        
        # Default validation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.validation_rules = {
            "input_text": InputValidationRule(
                field_name="input_text",
                required=True,
                min_length=10,
                max_length=2000
            ),
            "quality": InputValidationRule(
                field_name="quality",
                required=True,
                allowed_values=["low", "medium", "high", "ultra"]
            ),
            "duration": InputValidationRule(
                field_name="duration",
                required=True,
                min_value=5,
                max_value=300
            ),
            "output_format": InputValidationRule(
                field_name="output_format",
                required=True,
                allowed_values=["mp4", "avi", "mov", "webm"]
            ),
            "uploaded_file": InputValidationRule(
                field_name="uploaded_file",
                required=False,
                file_types=["mp4", "avi", "mov", "webm", "jpg", "png", "gif"],
                max_file_size=100 * 1024 * 1024  # 100MB
            )
        }
    
    def validate_text_input(
        self,
        text: str,
        field_name: str = "input_text",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate text input."""
        if not text or not text.strip():
            raise ValidationError(f"{field_name} is required")
        
        text = text.strip()
        rule = self.validation_rules.get(field_name, InputValidationRule(field_name=field_name))
        
        if rule.min_length and len(text) < rule.min_length:
            raise ValidationError(
                f"{field_name} must be at least {rule.min_length} characters long"
            )
        
        if rule.max_length and len(text) > rule.max_length:
            raise ValidationError(
                f"{field_name} must be at most {rule.max_length} characters long"
            )
        
        # Custom validation
        if rule.custom_validator:
            rule.custom_validator(text, context)
    
    def validate_numeric_input(
        self,
        value: Union[int, float],
        field_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate numeric input."""
        rule = self.validation_rules.get(field_name, InputValidationRule(field_name=field_name))
        
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(
                f"{field_name} must be at least {rule.min_value}"
            )
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(
                f"{field_name} must be at most {rule.max_value}"
            )
        
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(map(str, rule.allowed_values))}"
            )
    
    def validate_file_input(
        self,
        file_path: Optional[str],
        field_name: str = "uploaded_file",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate file input."""
        if not file_path:
            return  # Optional file
        
        rule = self.validation_rules.get(field_name, InputValidationRule(field_name=field_name))
        
        # Check if file exists
        if not Path(file_path).exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Check file type
        if rule.file_types:
            file_extension = Path(file_path).suffix.lower().lstrip('.')
            if file_extension not in rule.file_types:
                raise ValidationError(
                    f"File type not supported. Allowed types: {', '.join(rule.file_types)}"
                )
        
        # Check file size
        if rule.max_file_size:
            file_size = Path(file_path).stat().st_size
            if file_size > rule.max_file_size:
                max_size_mb = rule.max_file_size / (1024 * 1024)
                raise ValidationError(
                    f"File too large. Maximum size: {max_size_mb:.1f} MB"
                )
    
    async def validate_video_request(
        self,
        input_text: str,
        quality: str,
        duration: int,
        output_format: str,
        uploaded_file: Optional[str] = None,
        **kwargs
    ) -> VideoRequest:
        """Validate and create VideoRequest."""
        try:
            # Validate text input
            self.validate_text_input(input_text, "input_text")
            
            # Validate quality
            self.validate_numeric_input(quality, "quality", {"allowed_values": ["low", "medium", "high", "ultra"]})
            
            # Validate duration
            self.validate_numeric_input(duration, "duration", {"min_value": 5, "max_value": 300})
            
            # Validate output format
            self.validate_numeric_input(output_format, "output_format", {"allowed_values": ["mp4", "avi", "mov", "webm"]})
            
            # Validate file if provided
            if uploaded_file:
                self.validate_file_input(uploaded_file, "uploaded_file")
            
            # Create VideoRequest
            return VideoRequest(
                input_text=input_text,
                quality=quality,
                duration=duration,
                output_format=output_format,
                user_id=kwargs.get("user_id", "gradio_user"),
                request_id=kwargs.get("request_id", f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                **kwargs
            )
            
        except ValidationError as e:
            # Re-raise with context for better error handling
            raise ValidationError(f"Input validation failed: {str(e)}")


def gradio_error_handler(
    show_technical: bool = False,
    log_errors: bool = True
):
    """
    Decorator for handling errors in Gradio functions.
    
    Args:
        show_technical: Whether to show technical error details
        log_errors: Whether to log errors
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            error_handler = GradioErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Format error for Gradio
                title, description, details = error_handler.format_error_for_gradio(
                    e, show_technical=show_technical
                )
                
                # Log error if requested
                if log_errors:
                    logger.error(f"Gradio function error in {func.__name__}: {e}", exc_info=True)
                
                # Return error information
                return {
                    "error": True,
                    "title": title,
                    "description": description,
                    "details": details,
                    "success": False
                }
        
        return wrapper
    return decorator


def gradio_input_validator(
    validation_rules: Optional[Dict[str, InputValidationRule]] = None
):
    """
    Decorator for input validation in Gradio functions.
    
    Args:
        validation_rules: Custom validation rules
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            error_handler = GradioErrorHandler()
            validator = GradioInputValidator(error_handler)
            
            # Apply custom validation rules
            if validation_rules:
                validator.validation_rules.update(validation_rules)
            
            try:
                # Extract function parameters
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Validate inputs based on function signature
                for param_name, param_value in bound_args.arguments.items():
                    if param_name in validator.validation_rules:
                        rule = validator.validation_rules[param_name]
                        
                        if rule.required and (param_value is None or param_value == ""):
                            raise ValidationError(f"{param_name} is required")
                        
                        if param_value is not None and param_value != "":
                            if isinstance(param_value, str):
                                validator.validate_text_input(param_value, param_name)
                            elif isinstance(param_value, (int, float)):
                                validator.validate_numeric_input(param_value, param_name)
                            elif isinstance(param_value, str) and Path(param_value).exists():
                                validator.validate_file_input(param_value, param_name)
                
                return func(*args, **kwargs)
                
            except ValidationError as e:
                # Format validation error
                title, description, details = error_handler.format_error_for_gradio(
                    e, show_technical=False
                )
                
                return {
                    "error": True,
                    "title": title,
                    "description": description,
                    "details": details,
                    "success": False
                }
        
        return wrapper
    return decorator


def create_gradio_error_components() -> Tuple[gr.HTML, gr.HTML, gr.HTML]:
    """Create Gradio components for error display."""
    error_title = gr.HTML(
        value="",
        elem_classes=["error-title"],
        visible=False
    )
    
    error_description = gr.HTML(
        value="",
        elem_classes=["error-description"],
        visible=False
    )
    
    error_details = gr.HTML(
        value="",
        elem_classes=["error-details"],
        visible=False
    )
    
    return error_title, error_description, error_details


def update_error_display(
    result: Dict[str, Any],
    error_title: gr.HTML,
    error_description: gr.HTML,
    error_details: gr.HTML
) -> Tuple[gr.HTML, gr.HTML, gr.HTML, bool]:
    """Update error display components."""
    if result.get("error", False):
        # Show error components
        return (
            gr.HTML(value=result["title"], visible=True),
            gr.HTML(value=result["description"], visible=True),
            gr.HTML(value=result["details"], visible=True),
            False  # Hide success message
        )
    else:
        # Hide error components
        return (
            gr.HTML(value="", visible=False),
            gr.HTML(value="", visible=False),
            gr.HTML(value="", visible=False),
            True  # Show success message
        )


# Global error handler instance
gradio_error_handler_instance = GradioErrorHandler()


def handle_gradio_error(
    error: Exception,
    show_technical: bool = False
) -> Dict[str, Any]:
    """Handle error and return formatted response for Gradio."""
    title, description, details = gradio_error_handler_instance.format_error_for_gradio(
        error, show_technical=show_technical
    )
    
    return {
        "error": True,
        "title": title,
        "description": description,
        "details": details,
        "success": False
    }


def validate_gradio_inputs(
    input_text: str,
    quality: str,
    duration: int,
    output_format: str,
    uploaded_file: Optional[str] = None,
    **kwargs
) -> Union[VideoRequest, Dict[str, Any]]:
    """Validate Gradio inputs and return VideoRequest or error."""
    try:
        validator = GradioInputValidator(gradio_error_handler_instance)
        return validator.validate_video_request(
            input_text=input_text,
            quality=quality,
            duration=duration,
            output_format=output_format,
            uploaded_file=uploaded_file,
            **kwargs
        )
    except ValidationError as e:
        return handle_gradio_error(e, show_technical=False) 