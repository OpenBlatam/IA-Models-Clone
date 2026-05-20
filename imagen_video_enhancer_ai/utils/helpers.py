"""
Additional Helper Utilities for Imagen Video Enhancer AI
=======================================================

Various utility functions for common operations.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_id(prefix: str = "task") -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: ID prefix
        
    Returns:
        Unique ID string
    """
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.1f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m"


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded data or default
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        
    Returns:
        True if successful
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logger.error(f"Could not save JSON to {file_path}: {e}")
        return False


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(file_path: str) -> str:
    """
    Get file extension (without dot).
    
    Args:
        file_path: File path
        
    Returns:
        Extension string
    """
    return Path(file_path).suffix.lstrip(".").lower()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem[:250], Path(filename).suffix
        filename = name + ext
    return filename


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Decorator for retrying functions on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        
    Usage:
        @retry_on_exception(max_retries=3)
        def my_function():
            ...
    """
    def decorator(func):
        import functools
        import asyncio
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper
    
    return decorator




