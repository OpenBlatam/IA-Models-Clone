"""
Unified Storage System for Color Grading AI
============================================

Unified storage system consolidating local file operations and cloud storage.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import shutil

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Storage types."""
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


@dataclass
class StorageMetadata:
    """Storage metadata."""
    path: str
    size: int
    content_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    async def upload(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Upload file."""
        pass
    
    @abstractmethod
    async def download(self, remote_path: str, local_path: Union[str, Path]) -> str:
        """Download file."""
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files."""
        pass
    
    @abstractmethod
    async def get_metadata(self, path: str) -> Optional[StorageMetadata]:
        """Get file metadata."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local file system storage backend."""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize local storage.
        
        Args:
            base_path: Base storage path
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Copy file to storage."""
        source = Path(local_path)
        dest = self.base_path / remote_path
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        
        logger.debug(f"Uploaded {local_path} to {dest}")
        return str(dest)
    
    async def download(self, remote_path: str, local_path: Union[str, Path]) -> str:
        """Copy file from storage."""
        source = self.base_path / remote_path
        dest = Path(local_path)
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        
        logger.debug(f"Downloaded {remote_path} to {local_path}")
        return str(dest)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return (self.base_path / path).exists()
    
    async def delete(self, path: str) -> bool:
        """Delete file."""
        file_path = self.base_path / path
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            else:
                shutil.rmtree(file_path)
            logger.debug(f"Deleted {path}")
            return True
        return False
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files."""
        search_path = self.base_path / prefix if prefix else self.base_path
        
        if not search_path.exists():
            return []
        
        files = []
        for item in search_path.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(self.base_path)
                files.append(str(relative_path))
        
        return files
    
    async def get_metadata(self, path: str) -> Optional[StorageMetadata]:
        """Get file metadata."""
        file_path = self.base_path / path
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        
        return StorageMetadata(
            path=path,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime)
        )


class UnifiedStorage:
    """
    Unified storage system.
    
    Features:
    - Local and cloud storage
    - Automatic backend selection
    - Hybrid storage
    - Metadata management
    - File operations
    - Path normalization
    """
    
    def __init__(
        self,
        local_backend: Optional[LocalStorageBackend] = None,
        cloud_backend: Optional[StorageBackend] = None,
        default_storage: StorageType = StorageType.LOCAL
    ):
        """
        Initialize unified storage.
        
        Args:
            local_backend: Local storage backend
            cloud_backend: Cloud storage backend
            default_storage: Default storage type
        """
        self.local_backend = local_backend
        self.cloud_backend = cloud_backend
        self.default_storage = default_storage
        self._metadata_cache: Dict[str, StorageMetadata] = {}
    
    async def upload(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        storage_type: Optional[StorageType] = None
    ) -> str:
        """
        Upload file.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            storage_type: Optional storage type
            
        Returns:
            Storage path
        """
        storage_type = storage_type or self.default_storage
        
        if storage_type == StorageType.LOCAL:
            if not self.local_backend:
                raise ValueError("Local backend not configured")
            return await self.local_backend.upload(local_path, remote_path)
        
        elif storage_type == StorageType.CLOUD:
            if not self.cloud_backend:
                raise ValueError("Cloud backend not configured")
            return await self.cloud_backend.upload(local_path, remote_path)
        
        elif storage_type == StorageType.HYBRID:
            # Upload to both
            local_result = None
            cloud_result = None
            
            if self.local_backend:
                local_result = await self.local_backend.upload(local_path, remote_path)
            
            if self.cloud_backend:
                cloud_result = await self.cloud_backend.upload(local_path, remote_path)
            
            return cloud_result or local_result or remote_path
        
        raise ValueError(f"Invalid storage type: {storage_type}")
    
    async def download(
        self,
        remote_path: str,
        local_path: Union[str, Path],
        storage_type: Optional[StorageType] = None
    ) -> str:
        """
        Download file.
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            storage_type: Optional storage type
            
        Returns:
            Local file path
        """
        storage_type = storage_type or self.default_storage
        
        if storage_type == StorageType.LOCAL:
            if not self.local_backend:
                raise ValueError("Local backend not configured")
            return await self.local_backend.download(remote_path, local_path)
        
        elif storage_type == StorageType.CLOUD:
            if not self.cloud_backend:
                raise ValueError("Cloud backend not configured")
            return await self.cloud_backend.download(remote_path, local_path)
        
        elif storage_type == StorageType.HYBRID:
            # Try local first, then cloud
            if self.local_backend and await self.local_backend.exists(remote_path):
                return await self.local_backend.download(remote_path, local_path)
            elif self.cloud_backend:
                return await self.cloud_backend.download(remote_path, local_path)
            else:
                raise FileNotFoundError(f"File not found: {remote_path}")
        
        raise ValueError(f"Invalid storage type: {storage_type}")
    
    async def exists(
        self,
        path: str,
        storage_type: Optional[StorageType] = None
    ) -> bool:
        """Check if file exists."""
        storage_type = storage_type or self.default_storage
        
        if storage_type == StorageType.LOCAL:
            return await self.local_backend.exists(path) if self.local_backend else False
        elif storage_type == StorageType.CLOUD:
            return await self.cloud_backend.exists(path) if self.cloud_backend else False
        elif storage_type == StorageType.HYBRID:
            local_exists = await self.local_backend.exists(path) if self.local_backend else False
            cloud_exists = await self.cloud_backend.exists(path) if self.cloud_backend else False
            return local_exists or cloud_exists
        
        return False
    
    async def delete(
        self,
        path: str,
        storage_type: Optional[StorageType] = None
    ) -> bool:
        """Delete file."""
        storage_type = storage_type or self.default_storage
        
        results = []
        
        if storage_type in [StorageType.LOCAL, StorageType.HYBRID] and self.local_backend:
            results.append(await self.local_backend.delete(path))
        
        if storage_type in [StorageType.CLOUD, StorageType.HYBRID] and self.cloud_backend:
            results.append(await self.cloud_backend.delete(path))
        
        return any(results)
    
    async def list_files(
        self,
        prefix: str = "",
        storage_type: Optional[StorageType] = None
    ) -> List[str]:
        """List files."""
        storage_type = storage_type or self.default_storage
        
        if storage_type == StorageType.LOCAL:
            return await self.local_backend.list_files(prefix) if self.local_backend else []
        elif storage_type == StorageType.CLOUD:
            return await self.cloud_backend.list_files(prefix) if self.cloud_backend else []
        elif storage_type == StorageType.HYBRID:
            local_files = await self.local_backend.list_files(prefix) if self.local_backend else []
            cloud_files = await self.cloud_backend.list_files(prefix) if self.cloud_backend else []
            # Merge and deduplicate
            return list(set(local_files + cloud_files))
        
        return []
    
    async def get_metadata(
        self,
        path: str,
        storage_type: Optional[StorageType] = None
    ) -> Optional[StorageMetadata]:
        """Get file metadata."""
        # Check cache
        cache_key = f"{storage_type or self.default_storage}:{path}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        storage_type = storage_type or self.default_storage
        
        metadata = None
        
        if storage_type == StorageType.LOCAL and self.local_backend:
            metadata = await self.local_backend.get_metadata(path)
        elif storage_type == StorageType.CLOUD and self.cloud_backend:
            metadata = await self.cloud_backend.get_metadata(path)
        elif storage_type == StorageType.HYBRID:
            # Try local first
            if self.local_backend:
                metadata = await self.local_backend.get_metadata(path)
            if not metadata and self.cloud_backend:
                metadata = await self.cloud_backend.get_metadata(path)
        
        if metadata:
            self._metadata_cache[cache_key] = metadata
        
        return metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "local_backend": self.local_backend is not None,
            "cloud_backend": self.cloud_backend is not None,
            "default_storage": self.default_storage.value,
            "metadata_cache_size": len(self._metadata_cache),
        }

