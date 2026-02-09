"""
Common Helper Functions for Piel Mejorador AI SAM3
==================================================

This module provides common utility functions used throughout the application.
All functions are type-hinted and include proper error handling.

Note: This module now delegates to consolidated_helpers for better organization.
"""

from .consolidated_helpers import (
    create_message,
    load_json_file,
    save_json_file,
    ensure_directory_exists,
    create_output_directories,
    get_mime_type,
    FileOperations,
    MessageBuilder,
    DirectoryManager,
)

__all__ = [
    "create_message",
    "load_json_file",
    "save_json_file",
    "ensure_directory_exists",
    "create_output_directories",
    "get_mime_type",
    "FileOperations",
    "MessageBuilder",
    "DirectoryManager",
]


# Validation functions moved to validators.py
# These are kept for backward compatibility but delegate to validators
try:
    from .validators import FileValidator
except ImportError:
    # Fallback if validators not available
    FileValidator = None

# Validation functions - delegate to validators
try:
    from .validators import FileValidator
    
    def validate_image_file(file_path: str, max_size_mb: int = 50) -> bool:
        """Validate image file (backward compatibility)."""
        FileValidator.validate_image_file(file_path, max_size_mb)
        return True
    
    def validate_video_file(file_path: str, max_size_mb: int = 500) -> bool:
        """Validate video file (backward compatibility)."""
        FileValidator.validate_video_file(file_path, max_size_mb)
        return True
except ImportError:
    # Fallback if validators not available
    from pathlib import Path
    
    def validate_image_file(file_path: str, max_size_mb: int = 50) -> bool:
        """Validate image file (fallback)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"Image file too large: {size_mb:.2f}MB")
        return True
    
    def validate_video_file(file_path: str, max_size_mb: int = 500) -> bool:
        """Validate video file (fallback)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"Video file too large: {size_mb:.2f}MB")
        return True

