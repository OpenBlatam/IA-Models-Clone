"""File and path utilities."""

from pathlib import Path
from typing import Optional, List
import hashlib
import mimetypes
import os
import aiofiles

from utils.logger import get_logger

logger = get_logger(__name__)


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Get file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


async def get_file_hash_async(file_path: Path, algorithm: str = "md5") -> str:
    """
    Get file hash (async).
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(4096)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return file_path.stat().st_size if file_path.exists() else 0


def get_file_mime_type(file_path: Path) -> Optional[str]:
    """
    Get MIME type of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type or None
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: Path) -> Optional[str]:
    """
    Get file extension (without dot).
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension or None
    """
    return file_path.suffix.lstrip('.').lower() if file_path.suffix else None


def is_image_file(file_path: Path) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if image file
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    return file_path.suffix.lower() in image_extensions


def get_directory_size(directory: Path) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                pass
    
    return total_size


def list_files(
    directory: Path,
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory.
    
    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.jpg")
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    if not directory.exists():
        return []
    
    if pattern:
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    else:
        if recursive:
            return [p for p in directory.rglob("*") if p.is_file()]
        else:
            return [p for p in directory.iterdir() if p.is_file()]


def safe_delete(file_path: Path) -> bool:
    """
    Safely delete file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def copy_file(source: Path, destination: Path) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if copied successfully
    """
    try:
        import shutil
        ensure_directory(destination.parent)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False


def move_file(source: Path, destination: Path) -> bool:
    """
    Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if moved successfully
    """
    try:
        ensure_directory(destination.parent)
        source.rename(destination)
        return True
    except Exception as e:
        logger.error(f"Error moving file: {e}")
        return False

