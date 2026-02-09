"""
Error Handling Module

Provides:
- Custom exceptions
- Error handling utilities
- Error recovery
"""

from .exceptions import (
    ModelError,
    TrainingError,
    InferenceError,
    ValidationError,
    ConfigurationError
)

from .error_handler import (
    ErrorHandler,
    handle_error,
    retry_on_error
)

__all__ = [
    # Exceptions
    "ModelError",
    "TrainingError",
    "InferenceError",
    "ValidationError",
    "ConfigurationError",
    # Error handling
    "ErrorHandler",
    "handle_error",
    "retry_on_error"
]



