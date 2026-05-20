"""
Utility functions for video generation services
"""

from .error_handler import (
    VideoGenerationError,
    ImageGenerationError,
    AudioGenerationError,
    VideoCompositionError,
    retry_on_failure,
    handle_ffmpeg_error,
    validate_file_path,
    validate_image_file,
    validate_audio_file,
    safe_divide,
)

__all__ = [
    "VideoGenerationError",
    "ImageGenerationError",
    "AudioGenerationError",
    "VideoCompositionError",
    "retry_on_failure",
    "handle_ffmpeg_error",
    "validate_file_path",
    "validate_image_file",
    "validate_audio_file",
    "safe_divide",
]

