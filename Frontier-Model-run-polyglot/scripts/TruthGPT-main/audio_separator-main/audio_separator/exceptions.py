"""
Custom exceptions for audio separator.
"""

from typing import Optional, Dict, Any


class AudioSeparatorError(Exception):
    """
    Base exception for all audio separator errors.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.component = component
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.component:
            parts.append(f"Component: {self.component}")
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        return " | ".join(parts)


class AudioProcessingError(AudioSeparatorError):
    """Error during audio processing."""
    pass


class AudioFormatError(AudioSeparatorError):
    """Error related to unsupported or invalid audio formats."""
    pass


class AudioModelError(AudioSeparatorError):
    """Error related to AI models for audio separation."""
    pass


class AudioValidationError(AudioSeparatorError):
    """Error in data or parameter validation."""
    pass


class AudioIOError(AudioSeparatorError):
    """Error in audio file read/write operations."""
    pass


class AudioInitializationError(AudioSeparatorError):
    """Error during component initialization."""
    pass


class AudioConfigurationError(AudioSeparatorError):
    """Error in configuration."""
    pass

