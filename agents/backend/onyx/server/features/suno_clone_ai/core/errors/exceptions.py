"""
Custom Exceptions

Custom exception classes for better error handling.
"""


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class TrainingError(Exception):
    """Exception for training-related errors."""
    pass


class InferenceError(Exception):
    """Exception for inference-related errors."""
    pass


class ValidationError(Exception):
    """Exception for validation-related errors."""
    pass


class ConfigurationError(Exception):
    """Exception for configuration-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Exception when model is not found."""
    pass


class ModelLoadError(ModelError):
    """Exception when model loading fails."""
    pass


class ModelSaveError(ModelError):
    """Exception when model saving fails."""
    pass


class TrainingConvergenceError(TrainingError):
    """Exception when training fails to converge."""
    pass


class GradientExplosionError(TrainingError):
    """Exception when gradients explode."""
    pass


class InferenceTimeoutError(InferenceError):
    """Exception when inference times out."""
    pass


class InvalidInputError(ValidationError):
    """Exception for invalid inputs."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Exception for invalid configurations."""
    pass



