"""
File Manager for Flux2 Clothing Changer
========================================

Advanced file management and storage system.
"""

import os
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information."""
    file_id: str
    file_path: Path
    file_name: str
    file_size: int
    file_hash: str
    mime_type: Optional[str] = None
    created_at: float = 0.0
    modified_at: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FileManager:
    """Advanced file management system."""
    
    def __init__(
        self,
        base_directory: Path = Path("storage"),
        enable_hashing: bool = True,
    ):
        """
        Initialize file manager.
        
        Args:
            base_directory: Base storage directory
            enable_hashing: Enable file hashing
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.enable_hashing = enable_hashing
        
        self.files: Dict[str, FileInfo] = {}
        self.file_index: Dict[str, List[str]] = defaultdict(list)
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate file hash."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def store_file(
        self,
        source_path: Path,
        file_name: Optional[str] = None,
        subdirectory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FileInfo:
        """
        Store file.
        
        Args:
            source_path: Source file path
            file_name: Optional file name
            subdirectory: Optional subdirectory
            metadata: Optional metadata
            
        Returns:
            File information
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        file_name = file_name or source_path.name
        file_id = hashlib.md5(f"{file_name}{os.path.getmtime(source_path)}".encode()).hexdigest()
        
        # Determine destination
        if subdirectory:
            dest_dir = self.base_directory / subdirectory
        else:
            dest_dir = self.base_directory
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_name
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Get file info
        file_size = dest_path.stat().st_size
        file_hash = self._calculate_hash(dest_path) if self.enable_hashing else ""
        created_at = dest_path.stat().st_ctime
        modified_at = dest_path.stat().st_mtime
        
        file_info = FileInfo(
            file_id=file_id,
            file_path=dest_path,
            file_name=file_name,
            file_size=file_size,
            file_hash=file_hash,
            created_at=created_at,
            modified_at=modified_at,
            metadata=metadata or {},
        )
        
        self.files[file_id] = file_info
        logger.info(f"Stored file: {file_id} ({file_name})")
        
        return file_info
    
    def get_file(self, file_id: str) -> Optional[FileInfo]:
        """Get file information."""
        return self.files.get(file_id)
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete file.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if deleted
        """
        if file_id not in self.files:
            return False
        
        file_info = self.files[file_id]
        
        try:
            if file_info.file_path.exists():
                file_info.file_path.unlink()
            del self.files[file_id]
            logger.info(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def list_files(
        self,
        subdirectory: Optional[str] = None,
    ) -> List[FileInfo]:
        """
        List files.
        
        Args:
            subdirectory: Optional subdirectory filter
            
        Returns:
            List of file information
        """
        if subdirectory:
            return [
                f for f in self.files.values()
                if subdirectory in str(f.file_path.relative_to(self.base_directory))
            ]
        return list(self.files.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file manager statistics."""
        total_size = sum(f.file_size for f in self.files.values())
        
        return {
            "total_files": len(self.files),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


