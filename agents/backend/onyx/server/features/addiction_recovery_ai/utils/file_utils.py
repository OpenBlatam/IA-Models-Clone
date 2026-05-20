"""
File utilities
File operation functions
"""

from typing import Optional, List, Dict, Any
import os
import json
import shutil
from pathlib import Path


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read file content
    
    Args:
        file_path: Path to file
        encoding: File encoding
    
    Returns:
        File content
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Write content to file
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def read_json(file_path: str) -> Any:
    """
    Read JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(file_path: str, data: Any, indent: int = 2) -> None:
    """
    Write data to JSON file
    
    Args:
        file_path: Path to JSON file
        data: Data to write
        indent: JSON indent
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def file_exists(file_path: str) -> bool:
    """
    Check if file exists
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file exists
    """
    return os.path.isfile(file_path)


def dir_exists(dir_path: str) -> bool:
    """
    Check if directory exists
    
    Args:
        dir_path: Path to directory
    
    Returns:
        True if directory exists
    """
    return os.path.isdir(dir_path)


def create_dir(dir_path: str, exist_ok: bool = True) -> None:
    """
    Create directory
    
    Args:
        dir_path: Path to directory
        exist_ok: Don't raise error if exists
    """
    os.makedirs(dir_path, exist_ok=exist_ok)


def delete_file(file_path: str) -> bool:
    """
    Delete file
    
    Args:
        file_path: Path to file
    
    Returns:
        True if deleted
    """
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def delete_dir(dir_path: str) -> bool:
    """
    Delete directory
    
    Args:
        dir_path: Path to directory
    
    Returns:
        True if deleted
    """
    try:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            return True
        return False
    except Exception:
        return False


def copy_file(source: str, destination: str) -> bool:
    """
    Copy file
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        True if copied
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        return True
    except Exception:
        return False


def move_file(source: str, destination: str) -> bool:
    """
    Move file
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        True if moved
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
        return True
    except Exception:
        return False


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_extension(file_path: str) -> str:
    """
    Get file extension
    
    Args:
        file_path: Path to file
    
    Returns:
        File extension
    """
    return Path(file_path).suffix


def get_file_name(file_path: str, with_extension: bool = True) -> str:
    """
    Get file name
    
    Args:
        file_path: Path to file
        with_extension: Include extension
    
    Returns:
        File name
    """
    path = Path(file_path)
    
    if with_extension:
        return path.name
    
    return path.stem


def list_files(dir_path: str, pattern: Optional[str] = None, recursive: bool = False) -> List[str]:
    """
    List files in directory
    
    Args:
        dir_path: Directory path
        pattern: File pattern (e.g., "*.txt")
        recursive: Recursive search
    
    Returns:
        List of file paths
    """
    path = Path(dir_path)
    
    if not path.exists():
        return []
    
    if recursive:
        if pattern:
            files = list(path.rglob(pattern))
        else:
            files = list(path.rglob("*"))
    else:
        if pattern:
            files = list(path.glob(pattern))
        else:
            files = list(path.glob("*"))
    
    return [str(f) for f in files if f.is_file()]


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
    
    Returns:
        File information dictionary
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }

