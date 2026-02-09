"""
Custom Exceptions
Domain-specific exceptions
"""


class RecoveryAIError(Exception):
    """Base exception for Recovery AI"""
    pass


class ModelError(RecoveryAIError):
    """Model-related errors"""
    pass


class ModelLoadError(ModelError):
    """Model loading error"""
    pass


class ModelInferenceError(ModelError):
    """Model inference error"""
    pass


class ModelTrainingError(ModelError):
    """Model training error"""
    pass


class DataError(RecoveryAIError):
    """Data-related errors"""
    pass


class DataValidationError(DataError):
    """Data validation error"""
    pass


class DataProcessingError(DataError):
    """Data processing error"""
    pass


class ConfigurationError(RecoveryAIError):
    """Configuration error"""
    pass


class InferenceError(RecoveryAIError):
    """Inference error"""
    pass


class CUDAOutOfMemoryError(InferenceError):
    """CUDA out of memory error"""
    pass


class ValidationError(RecoveryAIError):
    """Validation error"""
    pass













