"""
File Operations
==============

Common file operations utilities.
"""

import shutil
from pathlib import Path
from typing import Optional, Union


def save_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    create_dirs: bool = True
) -> Path:
    """
    Save file to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        Path to saved file
    """
    dest_path = Path(destination)
    
    if create_dirs:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(source, dest_path)
    return dest_path


def move_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    create_dirs: bool = True
) -> Path:
    """
    Move file to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        Path to moved file
    """
    dest_path = Path(destination)
    
    if create_dirs:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(source, dest_path)
    return dest_path


def delete_file(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: File path to delete
        missing_ok: Whether to ignore if file doesn't exist
        
    Returns:
        True if deleted, False otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        return missing_ok
    
    try:
        path.unlink()
        return True
    except Exception:
        return False


def ensure_file_exists(file_path: Union[str, Path], create_empty: bool = False) -> Path:
    """
    Ensure file exists.
    
    Args:
        file_path: File path
        create_empty: Whether to create empty file if it doesn't exist
        
    Returns:
        Path object
    """
    path = Path(file_path)
    
    if not path.exists():
        if create_empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            raise FileNotFoundError(f"File does not exist: {file_path}")
    
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: File path
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: File path
        
    Returns:
        File size in MB
    """
    return get_file_size(file_path) / (1024 * 1024)




