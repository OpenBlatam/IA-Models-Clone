"""
File Utilities
==============

File and path manipulation utilities.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
import mimetypes


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(path)


def get_file_extension(path: str) -> str:
    """
    Get file extension.
    
    Args:
        path: File path
        
    Returns:
        File extension (with dot)
    """
    return Path(path).suffix


def get_mime_type(path: str) -> Optional[str]:
    """
    Get MIME type of file.
    
    Args:
        path: File path
        
    Returns:
        MIME type or None
    """
    return mimetypes.guess_type(path)[0]


def is_image_file(path: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        path: File path
        
    Returns:
        True if image, False otherwise
    """
    mime_type = get_mime_type(path)
    if not mime_type:
        return False
    
    return mime_type.startswith('image/')


def safe_delete(path: str) -> bool:
    """
    Safely delete file or directory.
    
    Args:
        path: Path to delete
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        path_obj = Path(path)
        if path_obj.is_file():
            path_obj.unlink()
        elif path_obj.is_dir():
            shutil.rmtree(path_obj)
        return True
    except Exception:
        return False


def get_files_in_dir(
    directory: str,
    pattern: str = "*",
    recursive: bool = False
) -> List[str]:
    """
    Get list of files in directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.py")
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    path_obj = Path(directory)
    
    if recursive:
        files = path_obj.rglob(pattern)
    else:
        files = path_obj.glob(pattern)
    
    return [str(f) for f in files if f.is_file()]


def copy_file(source: str, destination: str) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if copied, False otherwise
    """
    try:
        shutil.copy2(source, destination)
        return True
    except Exception:
        return False


def move_file(source: str, destination: str) -> bool:
    """
    Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if moved, False otherwise
    """
    try:
        shutil.move(source, destination)
        return True
    except Exception:
        return False

