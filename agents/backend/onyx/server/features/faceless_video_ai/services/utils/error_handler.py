"""
Error handling utilities for video generation
"""

import logging
from typing import Optional, Callable, Any
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Base exception for video generation errors"""
    pass


class ImageGenerationError(VideoGenerationError):
    """Error during image generation"""
    pass


class AudioGenerationError(VideoGenerationError):
    """Error during audio generation"""
    pass


class VideoCompositionError(VideoGenerationError):
    """Error during video composition"""
    pass


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {str(e)}")
            
            raise last_exception
        
        return wrapper
    return decorator


def handle_ffmpeg_error(error: Exception, operation: str) -> VideoCompositionError:
    """Handle FFmpeg-specific errors"""
    error_msg = str(error)
    
    if "not found" in error_msg.lower() or "FileNotFoundError" in str(type(error)):
        return VideoCompositionError(
            f"FFmpeg not found. Please install FFmpeg to {operation}. "
            f"Visit https://ffmpeg.org/download.html"
        )
    elif "codec" in error_msg.lower():
        return VideoCompositionError(
            f"Codec error during {operation}: {error_msg}"
        )
    elif "permission" in error_msg.lower():
        return VideoCompositionError(
            f"Permission denied during {operation}: {error_msg}"
        )
    else:
        return VideoCompositionError(f"FFmpeg error during {operation}: {error_msg}")


def validate_file_path(file_path: str, file_type: str = "file") -> None:
    """Validate that a file path exists and is accessible"""
    from pathlib import Path
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"{file_type.capitalize()} not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if path.stat().st_size == 0:
        raise ValueError(f"{file_type.capitalize()} is empty: {file_path}")


def validate_image_file(image_path: str) -> None:
    """Validate that an image file is valid"""
    from PIL import Image
    
    validate_file_path(image_path, "image")
    
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {image_path}. Error: {str(e)}")


def validate_audio_file(audio_path: str) -> None:
    """Validate that an audio file is valid"""
    validate_file_path(audio_path, "audio")
    
    # Basic validation - check file extension
    valid_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.aac'}
    from pathlib import Path
    if Path(audio_path).suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported audio format: {audio_path}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator

