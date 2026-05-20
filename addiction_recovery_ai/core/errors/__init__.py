"""
Errors Module
Custom exceptions and error handling
"""

from .custom_exceptions import (
    RecoveryAIError,
    ModelError,
    ModelLoadError,
    ModelInferenceError,
    ModelTrainingError,
    DataError,
    DataValidationError,
    DataProcessingError,
    ConfigurationError,
    InferenceError,
    CUDAOutOfMemoryError,
    ValidationError
)

from .error_handler import (
    ErrorHandler,
    handle_errors,
    safe_inference
)

__all__ = [
    "RecoveryAIError",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "ModelTrainingError",
    "DataError",
    "DataValidationError",
    "DataProcessingError",
    "ConfigurationError",
    "InferenceError",
    "CUDAOutOfMemoryError",
    "ValidationError",
    "ErrorHandler",
    "handle_errors",
    "safe_inference"
]













