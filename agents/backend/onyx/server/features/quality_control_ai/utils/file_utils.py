"""
File Utilities

Utility functions for file operations.
"""

import os
from pathlib import Path
from typing import Optional, List
import mimetypes


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_extension(file_path: str) -> str:
    """
    Get file extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        File extension (without dot)
    """
    return Path(file_path).suffix.lstrip('.').lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if image file
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    return get_file_extension(file_path) in image_extensions


def get_mime_type(file_path: str) -> Optional[str]:
    """
    Get MIME type of file.
    
    Args:
        file_path: Path to file
    
    Returns:
        MIME type or None
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def list_files(
    directory: str,
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """
    List files in directory.
    
    Args:
        directory: Directory path
        pattern: Optional glob pattern
        recursive: Whether to search recursively
    
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    if pattern:
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
    else:
        if recursive:
            files = list(dir_path.rglob('*'))
        else:
            files = list(dir_path.glob('*'))
    
    # Filter out directories
    return [str(f) for f in files if f.is_file()]


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename.
    
    Args:
        filename: Original filename
        max_length: Maximum length
    
    Returns:
        Safe filename
    """
    from .string_utils import sanitize_filename
    
    # Sanitize
    safe = sanitize_filename(filename)
    
    # Truncate if needed
    if len(safe) > max_length:
        # Preserve extension
        ext = Path(safe).suffix
        name = Path(safe).stem
        max_name_length = max_length - len(ext)
        safe = name[:max_name_length] + ext
    
    return safe



