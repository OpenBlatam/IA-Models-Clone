"""
Validators for Piel Mejorador AI SAM3
=====================================

Enhanced validation for parameters and files.
"""

import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class ParameterValidator:
    """Validates parameters for skin enhancement."""
    
    VALID_ENHANCEMENT_LEVELS = ["low", "medium", "high", "ultra"]
    VALID_FILE_TYPES = ["image", "video"]
    
    @staticmethod
    def validate_enhancement_level(level: str) -> None:
        """Validate enhancement level."""
        if level not in ParameterValidator.VALID_ENHANCEMENT_LEVELS:
            raise ValidationError(
                f"Invalid enhancement level: {level}. "
                f"Must be one of: {ParameterValidator.VALID_ENHANCEMENT_LEVELS}"
            )
    
    @staticmethod
    def validate_realism_level(level: Optional[float]) -> None:
        """Validate realism level."""
        if level is not None:
            if not isinstance(level, (int, float)):
                raise ValidationError("Realism level must be a number")
            if not 0.0 <= level <= 1.0:
                raise ValidationError("Realism level must be between 0.0 and 1.0")
    
    @staticmethod
    def validate_file_type(file_type: str) -> None:
        """Validate file type."""
        if file_type not in ParameterValidator.VALID_FILE_TYPES:
            raise ValidationError(
                f"Invalid file type: {file_type}. "
                f"Must be one of: {ParameterValidator.VALID_FILE_TYPES}"
            )
    
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
    
    @staticmethod
    def validate_priority(priority: int) -> None:
        """Validate task priority."""
        if not isinstance(priority, int):
            raise ValidationError("Priority must be an integer")
        if priority < 0:
            raise ValidationError("Priority must be non-negative")
    
    @staticmethod
    def validate_custom_instructions(instructions: Optional[str]) -> None:
        """Validate custom instructions."""
        if instructions is not None:
            if not isinstance(instructions, str):
                raise ValidationError("Custom instructions must be a string")
            if len(instructions) > 1000:
                raise ValidationError("Custom instructions must be less than 1000 characters")


class FileValidator:
    """Validates file properties."""
    
    @staticmethod
    def validate_image_file(
        file_path: str,
        max_size_mb: int = 50,
        allowed_extensions: Optional[List[str]] = None
    ) -> None:
        """Validate image file."""
        if allowed_extensions is None:
            allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        
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
            allowed_extensions = [".mp4", ".mov", ".avi", ".webm"]
        
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




