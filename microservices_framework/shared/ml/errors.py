"""
Error Handling
Custom exceptions for ML operations with proper error messages.
"""


class MLServiceError(Exception):
    """Base exception for ML service errors."""
    pass


class ModelLoadError(MLServiceError):
    """Raised when model loading fails."""
    pass


class ModelNotFoundError(MLServiceError):
    """Raised when model is not found."""
    pass


class InferenceError(MLServiceError):
    """Raised when inference fails."""
    pass


class TrainingError(MLServiceError):
    """Raised when training fails."""
    pass


class ConfigurationError(MLServiceError):
    """Raised when configuration is invalid."""
    pass


class DataError(MLServiceError):
    """Raised when data processing fails."""
    pass


class GPUError(MLServiceError):
    """Raised when GPU operations fail."""
    pass



