"""
Custom Exceptions for Color Grading AI
======================================
"""


class ColorGradingError(Exception):
    """Base exception for color grading operations."""
    pass


class MediaNotFoundError(ColorGradingError):
    """Raised when media file is not found."""
    pass


class InvalidParametersError(ColorGradingError):
    """Raised when color parameters are invalid."""
    pass


class ProcessingError(ColorGradingError):
    """Raised when processing fails."""
    pass


class TemplateNotFoundError(ColorGradingError):
    """Raised when template is not found."""
    pass


class CacheError(ColorGradingError):
    """Raised when cache operation fails."""
    pass


class ExportError(ColorGradingError):
    """Raised when export operation fails."""
    pass




