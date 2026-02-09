"""
Path Utilities for Color Grading AI
====================================

Unified path management and utilities.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os
import shutil

logger = logging.getLogger(__name__)


class PathUtilities:
    """
    Unified path utilities.
    
    Features:
    - Path normalization
    - Directory management
    - File operations
    - Path validation
    - Relative path resolution
    - Extension handling
    """
    
    @staticmethod
    def normalize(path: Union[str, Path]) -> Path:
        """
        Normalize path.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized Path object
        """
        return Path(path).resolve()
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
        """
        Ensure parent directory exists.
        
        Args:
            file_path: File path
            
        Returns:
            Path object
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_remove(path: Union[str, Path]) -> bool:
        """
        Safely remove file or directory.
        
        Args:
            path: Path to remove
            
        Returns:
            True if successful
        """
        try:
            path = Path(path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")
            return False
    
    @staticmethod
    def get_extension(path: Union[str, Path]) -> str:
        """
        Get file extension.
        
        Args:
            path: File path
            
        Returns:
            Extension (with dot)
        """
        return Path(path).suffix
    
    @staticmethod
    def change_extension(path: Union[str, Path], new_extension: str) -> Path:
        """
        Change file extension.
        
        Args:
            path: File path
            new_extension: New extension (with or without dot)
            
        Returns:
            New Path object
        """
        path = Path(path)
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension
        return path.with_suffix(new_extension)
    
    @staticmethod
    def get_stem(path: Union[str, Path]) -> str:
        """
        Get file stem (name without extension).
        
        Args:
            path: File path
            
        Returns:
            Stem
        """
        return Path(path).stem
    
    @staticmethod
    def get_name(path: Union[str, Path]) -> str:
        """
        Get file name.
        
        Args:
            path: File path
            
        Returns:
            File name
        """
        return Path(path).name
    
    @staticmethod
    def is_absolute(path: Union[str, Path]) -> bool:
        """
        Check if path is absolute.
        
        Args:
            path: Path to check
            
        Returns:
            True if absolute
        """
        return Path(path).is_absolute()
    
    @staticmethod
    def make_relative(base: Union[str, Path], target: Union[str, Path]) -> Path:
        """
        Make path relative to base.
        
        Args:
            base: Base path
            target: Target path
            
        Returns:
            Relative path
        """
        base = Path(base).resolve()
        target = Path(target).resolve()
        try:
            return target.relative_to(base)
        except ValueError:
            # If not relative, return absolute
            return target
    
    @staticmethod
    def join(*parts: Union[str, Path]) -> Path:
        """
        Join path parts.
        
        Args:
            *parts: Path parts
            
        Returns:
            Joined Path object
        """
        result = Path(parts[0])
        for part in parts[1:]:
            result = result / part
        return result
    
    @staticmethod
    def validate_path(path: Union[str, Path], must_exist: bool = False) -> bool:
        """
        Validate path.
        
        Args:
            path: Path to validate
            must_exist: If True, path must exist
            
        Returns:
            True if valid
        """
        try:
            path = Path(path)
            if must_exist:
                return path.exists()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_size(path: Union[str, Path]) -> int:
        """
        Get file size.
        
        Args:
            path: File path
            
        Returns:
            Size in bytes
        """
        return Path(path).stat().st_size
    
    @staticmethod
    def is_video_file(path: Union[str, Path]) -> bool:
        """
        Check if file is a video.
        
        Args:
            path: File path
            
        Returns:
            True if video file
        """
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}
        return PathUtilities.get_extension(path).lower() in video_extensions
    
    @staticmethod
    def is_image_file(path: Union[str, Path]) -> bool:
        """
        Check if file is an image.
        
        Args:
            path: File path
            
        Returns:
            True if image file
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        return PathUtilities.get_extension(path).lower() in image_extensions
    
    @staticmethod
    def is_media_file(path: Union[str, Path]) -> bool:
        """
        Check if file is a media file (video or image).
        
        Args:
            path: File path
            
        Returns:
            True if media file
        """
        return PathUtilities.is_video_file(path) or PathUtilities.is_image_file(path)
    
    @staticmethod
    def create_output_structure(base_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, Path]:
        """
        Create output directory structure.
        
        Args:
            base_dir: Base directory
            subdirs: List of subdirectory names
            
        Returns:
            Dictionary mapping subdir names to Path objects
        """
        base = PathUtilities.ensure_dir(base_dir)
        structure = {}
        
        for subdir in subdirs:
            structure[subdir] = PathUtilities.ensure_dir(base / subdir)
        
        return structure
    
    @staticmethod
    def find_files(
        directory: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = True
    ) -> List[Path]:
        """
        Find files in directory.
        
        Args:
            directory: Directory to search
            pattern: Optional glob pattern
            recursive: If True, search recursively
            
        Returns:
            List of matching files
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        if pattern:
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))
        else:
            if recursive:
                return [f for f in directory.rglob("*") if f.is_file()]
            else:
                return [f for f in directory.glob("*") if f.is_file()]
    
    @staticmethod
    def get_unique_path(base_path: Union[str, Path], filename: str) -> Path:
        """
        Get unique path (adds number if file exists).
        
        Args:
            base_path: Base directory
            filename: Filename
            
        Returns:
            Unique Path object
        """
        base = Path(base_path)
        path = base / filename
        
        if not path.exists():
            return path
        
        stem = Path(filename).stem
        extension = Path(filename).suffix
        
        counter = 1
        while path.exists():
            new_filename = f"{stem}_{counter}{extension}"
            path = base / new_filename
            counter += 1
        
        return path




