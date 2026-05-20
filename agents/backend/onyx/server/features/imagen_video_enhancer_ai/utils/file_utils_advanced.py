"""
Advanced File Utilities
=======================

Advanced file operation utilities.
"""

import os
import shutil
import mimetypes
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FileUtilsAdvanced:
    """Advanced file utility functions."""
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: File path
            algorithm: Hash algorithm (sha256, sha512, md5)
            
        Returns:
            File hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get detailed file information.
        
        Args:
            file_path: File path
            
        Returns:
            File information dictionary
        """
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "size": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix,
            "mime_type": mimetypes.guess_type(str(path))[0],
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "exists": path.exists()
        }
    
    @staticmethod
    def ensure_directory(path: str, create: bool = True) -> bool:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
            create: If True, create if doesn't exist
            
        Returns:
            True if directory exists
        """
        dir_path = Path(path)
        if dir_path.exists() and dir_path.is_dir():
            return True
        
        if create:
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        
        return False
    
    @staticmethod
    def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
        """
        Copy file.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: If True, overwrite existing file
            
        Returns:
            True if successful
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source}")
            return False
        
        if dest_path.exists() and not overwrite:
            logger.warning(f"Destination file exists: {destination}")
            return False
        
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False
    
    @staticmethod
    def move_file(source: str, destination: str, overwrite: bool = False) -> bool:
        """
        Move file.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: If True, overwrite existing file
            
        Returns:
            True if successful
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source}")
            return False
        
        if dest_path.exists() and not overwrite:
            logger.warning(f"Destination file exists: {destination}")
            return False
        
        try:
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Moved {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        Delete file.
        
        Args:
            file_path: File path
            
        Returns:
            True if successful
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            logger.info(f"Deleted {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    @staticmethod
    def list_files(directory: str, pattern: Optional[str] = None, recursive: bool = False) -> List[str]:
        """
        List files in directory.
        
        Args:
            directory: Directory path
            pattern: Optional file pattern (e.g., "*.jpg")
            recursive: If True, search recursively
            
        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        if pattern:
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
        else:
            if recursive:
                files = [f for f in dir_path.rglob("*") if f.is_file()]
            else:
                files = [f for f in dir_path.glob("*") if f.is_file()]
        
        return [str(f) for f in files]
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Calculate total size of directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        total = 0
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return 0
        
        for path in dir_path.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        
        return total
    
    @staticmethod
    def clean_directory(directory: str, pattern: Optional[str] = None, older_than_days: Optional[int] = None) -> int:
        """
        Clean directory by removing files matching criteria.
        
        Args:
            directory: Directory path
            pattern: Optional file pattern
            older_than_days: Optional age in days
            
        Returns:
            Number of files deleted
        """
        deleted = 0
        files = FileUtilsAdvanced.list_files(directory, pattern, recursive=True)
        
        cutoff = None
        if older_than_days:
            cutoff = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        
        for file_path in files:
            path = Path(file_path)
            if cutoff and path.stat().st_mtime >= cutoff:
                continue
            
            if FileUtilsAdvanced.delete_file(file_path):
                deleted += 1
        
        return deleted




