"""
Input Validators
================

Robust input validation for API endpoints.
"""

import re
from typing import Optional, List
from pathlib import Path
from PIL import Image
import io

from .exceptions import ValidationError


def validate_category(category: str) -> str:
    """Validate category."""
    from ..config.settings import get_settings
    settings = get_settings()
    
    if category not in settings.supported_categories:
        raise ValidationError(
            f"Category '{category}' is not supported. Supported categories: {', '.join(settings.supported_categories)}",
            "INVALID_CATEGORY"
        )
    return category


def validate_text_input(text: str, max_length: int = 5000, min_length: int = 10) -> str:
    """Validate text input."""
    if not text or not isinstance(text, str):
        raise ValidationError("Text input is required", "MISSING_TEXT")
    
    text = text.strip()
    
    if len(text) < min_length:
        raise ValidationError(
            f"Text must be at least {min_length} characters long",
            "TEXT_TOO_SHORT"
        )
    
    if len(text) > max_length:
        raise ValidationError(
            f"Text must be at most {max_length} characters long",
            "TEXT_TOO_LONG"
        )
    
    # Check for potentially malicious content
    if re.search(r'<script|javascript:|onerror=|onload=', text, re.IGNORECASE):
        raise ValidationError("Text contains potentially unsafe content", "UNSAFE_CONTENT")
    
    return text


def validate_image_file(file_content: bytes, max_size_mb: int = 10) -> bytes:
    """Validate image file."""
    if not file_content:
        raise ValidationError("Image file is required", "MISSING_IMAGE")
    
    # Check file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValidationError(
            f"Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)",
            "IMAGE_TOO_LARGE"
        )
    
    # Validate image format
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        
        # Check dimensions
        image = Image.open(io.BytesIO(file_content))  # Reopen after verify
        width, height = image.size
        
        if width > 10000 or height > 10000:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too large",
                "IMAGE_DIMENSIONS_TOO_LARGE"
            )
        
        if width < 10 or height < 10:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too small",
                "IMAGE_DIMENSIONS_TOO_SMALL"
            )
        
        # Check format
        allowed_formats = ["JPEG", "PNG", "WEBP", "GIF"]
        if image.format not in allowed_formats:
            raise ValidationError(
                f"Image format '{image.format}' is not supported. Allowed formats: {', '.join(allowed_formats)}",
                "UNSUPPORTED_IMAGE_FORMAT"
            )
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Invalid image file: {str(e)}", "INVALID_IMAGE")
    
    return file_content


def validate_multiple_images(
    files: List[bytes], max_count: int = 5, max_size_mb: int = 10
) -> List[bytes]:
    """Validate multiple image files."""
    if not files:
        raise ValidationError("At least one image file is required", "MISSING_IMAGES")
    
    if len(files) > max_count:
        raise ValidationError(
            f"Too many images ({len(files)}). Maximum allowed: {max_count}",
            "TOO_MANY_IMAGES"
        )
    
    validated_files = []
    for i, file_content in enumerate(files):
        try:
            validated = validate_image_file(file_content, max_size_mb)
            validated_files.append(validated)
        except ValidationError as e:
            raise ValidationError(
                f"Image {i + 1} validation failed: {e.message}",
                e.error_code
            )
    
    return validated_files




