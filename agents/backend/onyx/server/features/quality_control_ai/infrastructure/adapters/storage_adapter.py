"""
Storage Adapter

Adapter for file storage operations.
"""

import logging
from typing import Optional, List, BinaryIO
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class StorageAdapter:
    """
    Adapter for file storage operations.
    
    Provides abstraction for storing and retrieving files.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize storage adapter.
        
        Args:
            base_path: Base path for storage
        """
        self.base_path = Path(base_path) if base_path else Path("./storage")
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageAdapter initialized: {self.base_path}")
    
    def save_file(
        self,
        file_data: bytes,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> str:
        """
        Save a file.
        
        Args:
            file_data: File data as bytes
            filename: Name of the file
            subdirectory: Optional subdirectory
        
        Returns:
            Path to saved file
        """
        try:
            target_dir = self.base_path
            if subdirectory:
                target_dir = target_dir / subdirectory
                target_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = target_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}", exc_info=True)
            raise
    
    def load_file(self, file_path: str) -> bytes:
        """
        Load a file.
        
        Args:
            file_path: Path to file
        
        Returns:
            File data as bytes
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_path / path
            
            with open(path, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Failed to load file: {str(e)}", exc_info=True)
            raise
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to file
        
        Returns:
            True if deleted, False if not found
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_path / path
            
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {path}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}", exc_info=True)
            return False
    
    def list_files(
        self,
        subdirectory: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """
        List files in storage.
        
        Args:
            subdirectory: Optional subdirectory
            pattern: Optional glob pattern
        
        Returns:
            List of file paths
        """
        try:
            target_dir = self.base_path
            if subdirectory:
                target_dir = target_dir / subdirectory
            
            if pattern:
                files = list(target_dir.glob(pattern))
            else:
                files = list(target_dir.rglob("*"))
            
            # Filter out directories
            files = [f for f in files if f.is_file()]
            
            return [str(f.relative_to(self.base_path)) for f in files]
        
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}", exc_info=True)
            return []



