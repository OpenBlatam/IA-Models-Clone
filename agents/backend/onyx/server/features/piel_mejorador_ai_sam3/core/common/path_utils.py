"""
Path Utilities for Piel Mejorador AI SAM3
==========================================

Unified path operations and utilities.
"""

import logging
from typing import Union, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PathUtils:
    """Unified path utilities."""
    
    @staticmethod
    def ensure_exists(path: Union[str, Path]) -> Path:
        """
        Ensure path exists (create if directory, ensure parent for file).
        
        Args:
            path: Path to ensure
            
        Returns:
            Path object
        """
        path_obj = Path(path)
        
        if path_obj.suffix:  # Has extension, likely a file
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        else:  # Directory
            path_obj.mkdir(parents=True, exist_ok=True)
        
        return path_obj
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def ensure_parent(path: Union[str, Path]) -> Path:
        """
        Ensure parent directory exists.
        
        Args:
            path: File path
            
        Returns:
            Path object
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def resolve(path: Union[str, Path]) -> Path:
        """
        Resolve path to absolute.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved Path
        """
        return Path(path).resolve()
    
    @staticmethod
    def is_safe(
        path: Union[str, Path],
        base_dir: Union[str, Path]
    ) -> bool:
        """
        Check if path is within base directory (prevents traversal).
        
        Args:
            path: Path to check
            base_dir: Base directory
            
        Returns:
            True if safe
        """
        try:
            resolved_path = Path(path).resolve()
            resolved_base = Path(base_dir).resolve()
            return str(resolved_path).startswith(str(resolved_base))
        except Exception:
            return False
    
    @staticmethod
    def safe_join(
        base_dir: Union[str, Path],
        *parts: str
    ) -> Optional[Path]:
        """
        Safely join paths within base directory.
        
        Args:
            base_dir: Base directory
            *parts: Path parts to join
            
        Returns:
            Safe path or None if outside base
        """
        base = Path(base_dir).resolve()
        joined = base
        
        for part in parts:
            # Normalize part
            part = part.lstrip('/').replace('..', '')
            joined = joined / part
        
        resolved = joined.resolve()
        
        if PathUtils.is_safe(resolved, base):
            return resolved
        
        return None
    
    @staticmethod
    def get_extension(path: Union[str, Path]) -> str:
        """
        Get file extension (lowercase, without dot).
        
        Args:
            path: File path
            
        Returns:
            Extension (e.g., "jpg", "mp4")
        """
        return Path(path).suffix.lstrip('.').lower()
    
    @staticmethod
    def get_name(path: Union[str, Path], with_extension: bool = True) -> str:
        """
        Get file or directory name.
        
        Args:
            path: Path
            with_extension: Whether to include extension
            
        Returns:
            Name
        """
        path_obj = Path(path)
        if with_extension:
            return path_obj.name
        return path_obj.stem
    
    @staticmethod
    def create_structure(
        base_dir: Union[str, Path],
        subdirs: List[str]
    ) -> dict[str, Path]:
        """
        Create directory structure.
        
        Args:
            base_dir: Base directory
            subdirs: List of subdirectory names
            
        Returns:
            Dictionary mapping subdir names to Path objects
        """
        base = PathUtils.ensure_dir(base_dir)
        structure = {}
        
        for subdir in subdirs:
            subdir_path = base / subdir
            PathUtils.ensure_dir(subdir_path)
            structure[subdir] = subdir_path
        
        return structure
    
    @staticmethod
    def glob_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """
        Find files matching pattern.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        
        if recursive:
            return list(dir_path.rglob(pattern))
        return list(dir_path.glob(pattern))
    
    @staticmethod
    def get_size(path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            path: File path
            
        Returns:
            Size in bytes
        """
        return Path(path).stat().st_size
    
    @staticmethod
    def get_size_mb(path: Union[str, Path]) -> float:
        """
        Get file size in megabytes.
        
        Args:
            path: File path
            
        Returns:
            Size in MB
        """
        return PathUtils.get_size(path) / (1024 * 1024)


# Convenience functions
def ensure_path(path: Union[str, Path]) -> Path:
    """Ensure path exists."""
    return PathUtils.ensure_exists(path)


def safe_path(path: Union[str, Path], base_dir: Union[str, Path]) -> bool:
    """Check if path is safe."""
    return PathUtils.is_safe(path, base_dir)


def get_file_extension(path: Union[str, Path]) -> str:
    """Get file extension."""
    return PathUtils.get_extension(path)




