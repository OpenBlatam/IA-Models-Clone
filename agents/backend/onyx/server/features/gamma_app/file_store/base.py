"""
File Store Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4


class StorageBackend(str, Enum):
    """Storage backends"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class StoredFile:
    """Stored file model"""
    
    def __init__(
        self,
        filename: str,
        file_path: str,
        file_size: int,
        content_type: str,
        storage_backend: StorageBackend
    ):
        self.id = str(uuid4())
        self.filename = filename
        self.file_path = file_path
        self.file_size = file_size
        self.content_type = content_type
        self.storage_backend = storage_backend
        self.created_at = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}


class FileStoreBase(ABC):
    """Base interface for file store"""
    
    @abstractmethod
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> StoredFile:
        """Upload file"""
        pass
    
    @abstractmethod
    async def download_file(self, file_id: str) -> bytes:
        """Download file"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        pass
    
    @abstractmethod
    async def get_file_url(self, file_id: str, expires_in: int = 3600) -> str:
        """Get file URL"""
        pass

