"""
Path Helper
Path manipulation utilities
"""

from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class PathHelper:
    """
    Path manipulation helper
    """
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_latest_file(
        directory: Union[str, Path],
        pattern: str = "*",
    ) -> Optional[Path]:
        """
        Get latest file in directory
        
        Args:
            directory: Directory path
            pattern: File pattern
            
        Returns:
            Latest file path or None
        """
        directory = Path(directory)
        if not directory.exists():
            return None
        
        files = list(directory.glob(pattern))
        if not files:
            return None
        
        return max(files, key=lambda p: p.stat().st_mtime)
    
    @staticmethod
    def create_unique_path(
        base_path: Union[str, Path],
        suffix: str = "",
    ) -> Path:
        """
        Create unique path by appending number if exists
        
        Args:
            base_path: Base path
            suffix: Suffix to add
            
        Returns:
            Unique path
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            return base_path
        
        stem = base_path.stem
        parent = base_path.parent
        extension = base_path.suffix
        
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}{extension}"
            if not new_path.exists():
                return new_path
            counter += 1
    
    @staticmethod
    def get_relative_path(
        path: Union[str, Path],
        base: Union[str, Path],
    ) -> Path:
        """
        Get relative path
        
        Args:
            path: Target path
            base: Base path
            
        Returns:
            Relative path
        """
        path = Path(path)
        base = Path(base)
        return path.relative_to(base)



