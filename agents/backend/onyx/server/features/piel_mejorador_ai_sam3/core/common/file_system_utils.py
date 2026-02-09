"""
File System Utilities for Piel Mejorador AI SAM3
================================================

Unified file system operations and utilities.
"""

import shutil
import logging
from typing import List, Optional, Callable, Union, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class FileSystemUtils:
    """Unified file system utilities."""
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory.
        
        Args:
            directory: Directory path
            pattern: Optional glob pattern (e.g., "*.jpg")
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        if pattern:
            if recursive:
                return list(dir_path.rglob(pattern))
            else:
                return list(dir_path.glob(pattern))
        else:
            if recursive:
                return [p for p in dir_path.rglob("*") if p.is_file()]
            else:
                return [p for p in dir_path.iterdir() if p.is_file()]
    
    @staticmethod
    def list_directories(
        directory: Union[str, Path],
        recursive: bool = False
    ) -> List[Path]:
        """
        List directories.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            
        Returns:
            List of directory paths
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        if recursive:
            return [p for p in dir_path.rglob("*") if p.is_dir()]
        else:
            return [p for p in dir_path.iterdir() if p.is_dir()]
    
    @staticmethod
    def find_files(
        directory: Union[str, Path],
        predicate: Callable[[Path], bool],
        recursive: bool = False
    ) -> List[Path]:
        """
        Find files matching predicate.
        
        Args:
            directory: Directory path
            predicate: Function to test files
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        files = FileSystemUtils.list_files(directory, recursive=recursive)
        return [f for f in files if predicate(f)]
    
    @staticmethod
    def filter_by_size(
        files: List[Path],
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> List[Path]:
        """
        Filter files by size.
        
        Args:
            files: List of file paths
            min_size: Minimum size in bytes
            max_size: Maximum size in bytes
            
        Returns:
            Filtered file paths
        """
        filtered = []
        for file_path in files:
            if not file_path.exists():
                continue
            
            size = file_path.stat().st_size
            
            if min_size and size < min_size:
                continue
            if max_size and size > max_size:
                continue
            
            filtered.append(file_path)
        
        return filtered
    
    @staticmethod
    def filter_by_extension(
        files: List[Path],
        extensions: List[str]
    ) -> List[Path]:
        """
        Filter files by extension.
        
        Args:
            files: List of file paths
            extensions: List of extensions (with or without dot)
            
        Returns:
            Filtered file paths
        """
        # Normalize extensions
        normalized = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        
        return [
            f for f in files
            if f.suffix.lower() in normalized
        ]
    
    @staticmethod
    def filter_by_age(
        files: List[Path],
        max_age: timedelta,
        reference_time: Optional[datetime] = None
    ) -> List[Path]:
        """
        Filter files by age.
        
        Args:
            files: List of file paths
            max_age: Maximum age
            reference_time: Reference time (defaults to now)
            
        Returns:
            Files older than max_age
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        cutoff = reference_time - max_age
        filtered = []
        
        for file_path in files:
            if not file_path.exists():
                continue
            
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                filtered.append(file_path)
        
        return filtered
    
    @staticmethod
    def get_directory_size(
        directory: Union[str, Path],
        recursive: bool = True
    ) -> int:
        """
        Get total size of directory.
        
        Args:
            directory: Directory path
            recursive: Whether to include subdirectories
            
        Returns:
            Total size in bytes
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        total_size = 0
        
        if recursive:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except Exception:
                        pass
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except Exception:
                        pass
        
        return total_size
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: File path
            
        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        if not path.exists():
            return {}
        
        stat = path.stat()
        
        return {
            "path": str(path.resolve()),
            "name": path.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "extension": path.suffix,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
    
    @staticmethod
    def delete_old_files(
        directory: Union[str, Path],
        max_age: timedelta,
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> int:
        """
        Delete old files.
        
        Args:
            directory: Directory path
            max_age: Maximum age
            pattern: Optional glob pattern
            recursive: Whether to search recursively
            
        Returns:
            Number of files deleted
        """
        files = FileSystemUtils.list_files(directory, pattern=pattern, recursive=recursive)
        old_files = FileSystemUtils.filter_by_age(files, max_age)
        
        deleted = 0
        for file_path in old_files:
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {e}")
        
        return deleted
    
    @staticmethod
    def cleanup_empty_directories(
        directory: Union[str, Path],
        recursive: bool = True
    ) -> int:
        """
        Remove empty directories.
        
        Args:
            directory: Directory path
            recursive: Whether to process recursively
            
        Returns:
            Number of directories removed
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        removed = 0
        
        if recursive:
            # Process from deepest to shallowest
            dirs = sorted(dir_path.rglob("*"), key=lambda p: len(p.parts), reverse=True)
            for d in dirs:
                if d.is_dir():
                    try:
                        d.rmdir()  # Only removes if empty
                        removed += 1
                    except OSError:
                        pass  # Directory not empty
        else:
            for d in dir_path.iterdir():
                if d.is_dir():
                    try:
                        d.rmdir()
                        removed += 1
                    except OSError:
                        pass
        
        return removed
    
    @staticmethod
    def copy_directory(
        source: Union[str, Path],
        destination: Union[str, Path],
        ignore_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Copy directory recursively.
        
        Args:
            source: Source directory
            destination: Destination directory
            ignore_patterns: Optional patterns to ignore
        """
        src = Path(source)
        dst = Path(destination)
        
        if not src.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")
        
        def ignore_func(directory, files):
            if ignore_patterns:
                ignored = []
                for pattern in ignore_patterns:
                    for file in files:
                        if pattern in file or Path(file).match(pattern):
                            ignored.append(file)
                return ignored
            return []
        
        shutil.copytree(src, dst, ignore=ignore_func if ignore_patterns else None, dirs_exist_ok=True)
    
    @staticmethod
    def move_directory(
        source: Union[str, Path],
        destination: Union[str, Path]
    ) -> None:
        """
        Move directory.
        
        Args:
            source: Source directory
            destination: Destination directory
        """
        src = Path(source)
        dst = Path(destination)
        
        if not src.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    
    @staticmethod
    def get_directory_tree(
        directory: Union[str, Path],
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get directory tree structure.
        
        Args:
            directory: Directory path
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary tree structure
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return {}
        
        def build_tree(path: Path, depth: int = 0) -> Dict[str, Any]:
            if max_depth and depth >= max_depth:
                return {}
            
            tree = {
                "name": path.name,
                "path": str(path),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
            }
            
            if path.is_file():
                try:
                    stat = path.stat()
                    tree["size"] = stat.st_size
                    tree["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                except Exception:
                    pass
            elif path.is_dir():
                tree["children"] = []
                try:
                    for child in sorted(path.iterdir()):
                        child_tree = build_tree(child, depth + 1)
                        if child_tree:
                            tree["children"].append(child_tree)
                except PermissionError:
                    pass
            
            return tree
        
        return build_tree(dir_path)


# Convenience functions
def list_files(directory: Union[str, Path], **kwargs) -> List[Path]:
    """List files in directory."""
    return FileSystemUtils.list_files(directory, **kwargs)


def find_files(directory: Union[str, Path], predicate: Callable[[Path], bool], **kwargs) -> List[Path]:
    """Find files matching predicate."""
    return FileSystemUtils.find_files(directory, predicate, **kwargs)


def get_directory_size(directory: Union[str, Path], **kwargs) -> int:
    """Get directory size."""
    return FileSystemUtils.get_directory_size(directory, **kwargs)


def delete_old_files(directory: Union[str, Path], max_age: timedelta, **kwargs) -> int:
    """Delete old files."""
    return FileSystemUtils.delete_old_files(directory, max_age, **kwargs)




