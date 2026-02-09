"""
Validators for Imagen Video Enhancer AI
=======================================
"""

import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class ParameterValidator:
    """Validates parameters for enhancement."""
    
    VALID_ENHANCEMENT_TYPES = ["general", "sharpness", "colors", "denoise", "upscale", "restore"]
    VALID_NOISE_LEVELS = ["low", "medium", "high"]
    VALID_CORRECTION_TYPES = ["auto", "manual", "vibrant", "natural"]
    
    @staticmethod
    def validate_enhancement_type(enhancement_type: str) -> None:
        """Validate enhancement type."""
        if enhancement_type not in ParameterValidator.VALID_ENHANCEMENT_TYPES:
            raise ValidationError(
                f"Invalid enhancement type: {enhancement_type}. "
                f"Must be one of: {ParameterValidator.VALID_ENHANCEMENT_TYPES}"
            )
    
    @staticmethod
    def validate_noise_level(noise_level: str) -> None:
        """Validate noise level."""
        if noise_level not in ParameterValidator.VALID_NOISE_LEVELS:
            raise ValidationError(
                f"Invalid noise level: {noise_level}. "
                f"Must be one of: {ParameterValidator.VALID_NOISE_LEVELS}"
            )
    
    @staticmethod
    def validate_scale_factor(scale_factor: int) -> None:
        """Validate scale factor."""
        if not isinstance(scale_factor, int):
            raise ValidationError("Scale factor must be an integer")
        if scale_factor < 1 or scale_factor > 8:
            raise ValidationError("Scale factor must be between 1 and 8")
    
    @staticmethod
    def validate_priority(priority: int) -> None:
        """Validate task priority."""
        if not isinstance(priority, int):
            raise ValidationError("Priority must be an integer")
        if priority < 0:
            raise ValidationError("Priority must be non-negative")
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> None:
        """Validate file path."""
        if not file_path:
            raise ValidationError("File path cannot be empty")
        
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if must_exist and not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")


class FileValidator:
    """Validates file properties."""
    
    @staticmethod
    def validate_image_file(
        file_path: str,
        max_size_mb: int = 100,
        allowed_extensions: Optional[List[str]] = None
    ) -> None:
        """Validate image file."""
        if allowed_extensions is None:
            allowed_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Image file not found: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check extension
        if path.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {allowed_extensions}"
            )
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"Image file too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
            )
    
    @staticmethod
    def validate_video_file(
        file_path: str,
        max_size_mb: int = 500,
        allowed_extensions: Optional[List[str]] = None
    ) -> None:
        """Validate video file."""
        if allowed_extensions is None:
            allowed_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Video file not found: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check extension
        if path.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported: {allowed_extensions}"
            )
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"Video file too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
            )
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Get file type (image or video) based on extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        
        if ext in image_extensions:
            return "image"
        elif ext in video_extensions:
            return "video"
        else:
            raise ValidationError(f"Unknown file type for extension: {ext}")




