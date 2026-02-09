"""Custom exceptions for the application."""


class PlasticSurgeryAIException(Exception):
    """Base exception for Plastic Surgery Visualization AI."""
    pass


class ImageProcessingError(PlasticSurgeryAIException):
    """Error during image processing."""
    pass


class ImageValidationError(PlasticSurgeryAIException):
    """Error during image validation."""
    pass


class AIProcessingError(PlasticSurgeryAIException):
    """Error during AI processing."""
    pass


class VisualizationNotFoundError(PlasticSurgeryAIException):
    """Visualization not found."""
    pass


class InvalidSurgeryTypeError(PlasticSurgeryAIException):
    """Invalid surgery type."""
    pass


class RateLimitExceededError(PlasticSurgeryAIException):
    """Rate limit exceeded."""
    pass


class StorageError(PlasticSurgeryAIException):
    """Error with storage operations."""
    pass


class ValidationError(PlasticSurgeryAIException):
    """Validation error."""
    pass

