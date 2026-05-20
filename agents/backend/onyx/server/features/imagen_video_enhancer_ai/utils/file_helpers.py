"""
File helpers for Imagen Video Enhancer AI
==========================================
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    
    # Fallback based on extension
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    
    return mime_map.get(ext, "application/octet-stream")


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def ensure_unique_filename(directory: Path, filename: str) -> Path:
    """
    Ensure filename is unique in directory.
    
    Args:
        directory: Target directory
        filename: Original filename
        
    Returns:
        Path with unique filename
    """
    path = directory / filename
    
    if not path.exists():
        return path
    
    # Add counter to filename
    stem = path.stem
    suffix = path.suffix
    counter = 1
    
    while path.exists():
        new_filename = f"{stem}_{counter}{suffix}"
        path = directory / new_filename
        counter += 1
    
    return path




