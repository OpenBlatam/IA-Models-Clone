"""
Custom Exceptions
Framework-specific exceptions
"""


class MLFrameworkError(Exception):
    """Base exception for ML framework"""
    pass


class ModelError(MLFrameworkError):
    """Exception for model-related errors"""
    pass


class TrainingError(MLFrameworkError):
    """Exception for training-related errors"""
    pass


class InferenceError(MLFrameworkError):
    """Exception for inference-related errors"""
    pass


class ConfigurationError(MLFrameworkError):
    """Exception for configuration-related errors"""
    pass


class ValidationError(MLFrameworkError):
    """Exception for validation errors"""
    pass


class DataError(MLFrameworkError):
    """Exception for data-related errors"""
    pass



