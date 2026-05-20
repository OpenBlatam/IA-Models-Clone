"""
Custom Exceptions
Domain-specific exceptions for better error handling
"""


class MLException(Exception):
    """Base exception for ML components"""
    pass


class ModelError(MLException):
    """Model-related errors"""
    pass


class TrainingError(MLException):
    """Training-related errors"""
    pass


class DataError(MLException):
    """Data-related errors"""
    pass


class ValidationError(MLException):
    """Validation errors"""
    pass


class ConfigurationError(MLException):
    """Configuration errors"""
    pass


class InferenceError(MLException):
    """Inference errors"""
    pass


class OptimizationError(MLException):
    """Optimization errors"""
    pass













