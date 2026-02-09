"""
Temporary File Utilities for Piel Mejorador AI SAM3
==================================================

Unified temporary file and directory management.
"""

import tempfile
import shutil
import logging
from typing import Optional, Union, Callable, Any
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager

logger = logging.getLogger(__name__)


class TempFileUtils:
    """Unified temporary file utilities."""
    
    @staticmethod
    @contextmanager
    def temp_file(
        suffix: str = "",
        prefix: str = "piel_mejorador_",
        delete: bool = True,
        mode: str = "w+b"
    ):
        """
        Create temporary file with automatic cleanup.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            delete: Whether to delete on exit
            mode: File mode
            
        Yields:
            Temporary file path
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        try:
            yield Path(path)
        finally:
            if delete:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error deleting temp file {path}: {e}")
    
    @staticmethod
    @contextmanager
    def temp_directory(
        prefix: str = "piel_mejorador_",
        delete: bool = True
    ):
        """
        Create temporary directory with automatic cleanup.
        
        Args:
            prefix: Directory prefix
            delete: Whether to delete on exit
            
        Yields:
            Temporary directory path
        """
        path = Path(tempfile.mkdtemp(prefix=prefix))
        try:
            yield path
        finally:
            if delete:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Error deleting temp directory {path}: {e}")
    
    @staticmethod
    def create_temp_file(
        suffix: str = "",
        prefix: str = "piel_mejorador_",
        content: Optional[bytes] = None
    ) -> Path:
        """
        Create temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Optional content to write
            
        Returns:
            Temporary file path
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        path_obj = Path(path)
        
        if content:
            path_obj.write_bytes(content)
        
        return path_obj
    
    @staticmethod
    def create_temp_directory(
        prefix: str = "piel_mejorador_"
    ) -> Path:
        """
        Create temporary directory.
        
        Args:
            prefix: Directory prefix
            
        Returns:
            Temporary directory path
        """
        return Path(tempfile.mkdtemp(prefix=prefix))
    
    @staticmethod
    def cleanup_temp_file(path: Union[str, Path]) -> bool:
        """
        Cleanup temporary file.
        
        Args:
            path: File path
            
        Returns:
            True if deleted
        """
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.warning(f"Error deleting temp file {path}: {e}")
            return False
    
    @staticmethod
    def cleanup_temp_directory(path: Union[str, Path]) -> bool:
        """
        Cleanup temporary directory.
        
        Args:
            path: Directory path
            
        Returns:
            True if deleted
        """
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception as e:
            logger.warning(f"Error deleting temp directory {path}: {e}")
            return False


class TempFileManager:
    """Manager for temporary files with tracking and cleanup."""
    
    def __init__(self, base_dir: Optional[Path] = None, auto_cleanup: bool = True):
        """
        Initialize temp file manager.
        
        Args:
            base_dir: Optional base directory for temp files
            auto_cleanup: Whether to cleanup on exit
        """
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = None
        
        self.auto_cleanup = auto_cleanup
        self._temp_files: list[Path] = []
        self._temp_dirs: list[Path] = []
    
    def create_file(
        self,
        suffix: str = "",
        prefix: str = "piel_mejorador_",
        content: Optional[bytes] = None
    ) -> Path:
        """
        Create temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Optional content
            
        Returns:
            Temporary file path
        """
        if self.base_dir:
            path = self.base_dir / f"{prefix}{tempfile.gettempprefix()}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if content:
                path.write_bytes(content)
        else:
            path = TempFileUtils.create_temp_file(suffix, prefix, content)
        
        self._temp_files.append(path)
        return path
    
    def create_directory(
        self,
        prefix: str = "piel_mejorador_"
    ) -> Path:
        """
        Create temporary directory.
        
        Args:
            prefix: Directory prefix
            
        Returns:
            Temporary directory path
        """
        if self.base_dir:
            path = self.base_dir / f"{prefix}{tempfile.gettempprefix()}"
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = TempFileUtils.create_temp_directory(prefix)
        
        self._temp_dirs.append(path)
        return path
    
    def cleanup_all(self) -> int:
        """
        Cleanup all temporary files and directories.
        
        Returns:
            Number of items cleaned up
        """
        count = 0
        
        for path in self._temp_files:
            if TempFileUtils.cleanup_temp_file(path):
                count += 1
        
        for path in self._temp_dirs:
            if TempFileUtils.cleanup_temp_directory(path):
                count += 1
        
        self._temp_files.clear()
        self._temp_dirs.clear()
        
        return count
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self.cleanup_all()


# Convenience functions
@contextmanager
def temp_file(**kwargs):
    """Create temporary file."""
    with TempFileUtils.temp_file(**kwargs) as path:
        yield path


@contextmanager
def temp_directory(**kwargs):
    """Create temporary directory."""
    with TempFileUtils.temp_directory(**kwargs) as path:
        yield path




