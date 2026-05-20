"""
File Service
============

Advanced file service with storage abstraction, compression, and security.
"""

from __future__ import annotations
import asyncio
import logging
import os
import hashlib
import mimetypes
import gzip
import zipfile
from typing import Dict, List, Optional, Any, Union, BinaryIO, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import aiohttp
from PIL import Image
import magic

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend enumeration"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    MINIO = "minio"


class FileType(str, Enum):
    """File type enumeration"""
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    CODE = "code"
    DATA = "data"
    UNKNOWN = "unknown"


class CompressionType(str, Enum):
    """Compression type enumeration"""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class FileMetadata:
    """File metadata representation"""
    filename: str
    original_filename: str
    file_type: FileType
    mime_type: str
    size: int
    checksum: str
    created_at: datetime
    modified_at: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Storage configuration"""
    backend: StorageBackend = StorageBackend.LOCAL
    base_path: str = "./storage"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".txt", ".doc", ".docx"])
    compression_enabled: bool = True
    compression_threshold: int = 1024 * 1024  # 1MB
    compression_type: CompressionType = CompressionType.GZIP
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    gcs_bucket: Optional[str] = None
    azure_container: Optional[str] = None


class FileService:
    """Advanced file service with multiple storage backends"""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._is_running = False
        self._storage_backend = None
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backend"""
        try:
            if self.config.backend == StorageBackend.LOCAL:
                # Create local storage directory
                Path(self.config.base_path).mkdir(parents=True, exist_ok=True)
                self._storage_backend = LocalStorage(self.config)
            
            elif self.config.backend == StorageBackend.S3:
                self._storage_backend = S3Storage(self.config)
            
            elif self.config.backend == StorageBackend.GCS:
                self._storage_backend = GCSStorage(self.config)
            
            elif self.config.backend == StorageBackend.AZURE:
                self._storage_backend = AzureStorage(self.config)
            
            elif self.config.backend == StorageBackend.MINIO:
                self._storage_backend = MinioStorage(self.config)
            
            else:
                raise ValueError(f"Unsupported storage backend: {self.config.backend}")
            
            logger.info(f"Storage backend initialized: {self.config.backend.value}")
        
        except Exception as e:
            logger.error(f"Failed to initialize storage backend: {e}")
            raise
    
    async def start(self):
        """Start the file service"""
        if self._is_running:
            return
        
        try:
            await self._storage_backend.initialize()
            self._is_running = True
            logger.info("File service started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start file service: {e}")
            raise
    
    async def stop(self):
        """Stop the file service"""
        if not self._is_running:
            return
        
        try:
            await self._storage_backend.cleanup()
            self._is_running = False
            logger.info("File service stopped")
        
        except Exception as e:
            logger.error(f"Error stopping file service: {e}")
    
    @measure_performance
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileMetadata:
        """Upload file to storage"""
        try:
            # Validate file
            await self._validate_file(file_content, filename)
            
            # Generate file metadata
            file_metadata = await self._generate_metadata(file_content, filename, metadata, tags)
            
            # Compress if needed
            if self.config.compression_enabled and len(file_content) > self.config.compression_threshold:
                file_content = await self._compress_file(file_content)
                file_metadata.metadata["compressed"] = True
                file_metadata.metadata["compression_type"] = self.config.compression_type.value
            
            # Encrypt if needed
            if self.config.encryption_enabled:
                file_content = await self._encrypt_file(file_content)
                file_metadata.metadata["encrypted"] = True
            
            # Upload to storage
            await self._storage_backend.upload(file_metadata.filename, file_content)
            
            logger.info(f"File uploaded successfully: {filename}")
            return file_metadata
        
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    @measure_performance
    async def download_file(self, filename: str) -> bytes:
        """Download file from storage"""
        try:
            # Download from storage
            file_content = await self._storage_backend.download(filename)
            
            # Decrypt if needed
            if self.config.encryption_enabled:
                file_content = await self._decrypt_file(file_content)
            
            # Decompress if needed
            if self.config.compression_enabled:
                file_content = await self._decompress_file(file_content)
            
            logger.info(f"File downloaded successfully: {filename}")
            return file_content
        
        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise
    
    async def delete_file(self, filename: str) -> bool:
        """Delete file from storage"""
        try:
            await self._storage_backend.delete(filename)
            logger.info(f"File deleted successfully: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False
    
    async def list_files(
        self,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        file_type: Optional[FileType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[FileMetadata]:
        """List files with filtering"""
        try:
            files = await self._storage_backend.list_files(prefix, limit, offset)
            
            # Filter by tags and file type
            filtered_files = []
            for file_metadata in files:
                if tags and not any(tag in file_metadata.tags for tag in tags):
                    continue
                
                if file_type and file_metadata.file_type != file_type:
                    continue
                
                filtered_files.append(file_metadata)
            
            return filtered_files
        
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            raise
    
    async def get_file_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        try:
            return await self._storage_backend.get_metadata(filename)
        
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return None
    
    async def update_file_metadata(
        self,
        filename: str,
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update file metadata"""
        try:
            await self._storage_backend.update_metadata(filename, metadata, tags)
            logger.info(f"File metadata updated: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update file metadata: {e}")
            return False
    
    async def _validate_file(self, file_content: bytes, filename: str):
        """Validate file"""
        # Check file size
        if len(file_content) > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed size: {self.config.max_file_size}")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_extensions:
            raise ValueError(f"File extension not allowed: {file_ext}")
        
        # Check file type
        mime_type = magic.from_buffer(file_content, mime=True)
        if not self._is_allowed_mime_type(mime_type):
            raise ValueError(f"MIME type not allowed: {mime_type}")
    
    def _is_allowed_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type is allowed"""
        allowed_mime_types = [
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "application/pdf", "text/plain", "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        return mime_type in allowed_mime_types
    
    async def _generate_metadata(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]]
    ) -> FileMetadata:
        """Generate file metadata"""
        # Generate checksum
        checksum = hashlib.sha256(file_content).hexdigest()
        
        # Detect file type
        mime_type = magic.from_buffer(file_content, mime=True)
        file_type = self._detect_file_type(mime_type)
        
        # Generate unique filename
        timestamp = int(DateTimeHelpers.now_utc().timestamp())
        unique_filename = f"{timestamp}_{filename}"
        
        return FileMetadata(
            filename=unique_filename,
            original_filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            size=len(file_content),
            checksum=checksum,
            created_at=DateTimeHelpers.now_utc(),
            modified_at=DateTimeHelpers.now_utc(),
            tags=tags or [],
            metadata=metadata or {}
        )
    
    def _detect_file_type(self, mime_type: str) -> FileType:
        """Detect file type from MIME type"""
        if mime_type.startswith("image/"):
            return FileType.IMAGE
        elif mime_type.startswith("video/"):
            return FileType.VIDEO
        elif mime_type.startswith("audio/"):
            return FileType.AUDIO
        elif mime_type in ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            return FileType.DOCUMENT
        elif mime_type in ["application/zip", "application/x-rar-compressed"]:
            return FileType.ARCHIVE
        elif mime_type.startswith("text/") or mime_type in ["application/json", "application/xml"]:
            return FileType.CODE
        else:
            return FileType.UNKNOWN
    
    async def _compress_file(self, file_content: bytes) -> bytes:
        """Compress file content"""
        if self.config.compression_type == CompressionType.GZIP:
            return gzip.compress(file_content)
        elif self.config.compression_type == CompressionType.ZIP:
            # For ZIP compression, we need to create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile() as temp_file:
                with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("file", file_content)
                temp_file.seek(0)
                return temp_file.read()
        else:
            return file_content
    
    async def _decompress_file(self, file_content: bytes) -> bytes:
        """Decompress file content"""
        if self.config.compression_type == CompressionType.GZIP:
            return gzip.decompress(file_content)
        elif self.config.compression_type == CompressionType.ZIP:
            import tempfile
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                with zipfile.ZipFile(temp_file, 'r') as zip_file:
                    return zip_file.read("file")
        else:
            return file_content
    
    async def _encrypt_file(self, file_content: bytes) -> bytes:
        """Encrypt file content"""
        # Simple XOR encryption (in production, use proper encryption)
        if not self.config.encryption_key:
            raise ValueError("Encryption key not configured")
        
        key = self.config.encryption_key.encode()
        encrypted = bytearray()
        for i, byte in enumerate(file_content):
            encrypted.append(byte ^ key[i % len(key)])
        
        return bytes(encrypted)
    
    async def _decrypt_file(self, file_content: bytes) -> bytes:
        """Decrypt file content"""
        # Simple XOR decryption (in production, use proper decryption)
        if not self.config.encryption_key:
            raise ValueError("Encryption key not configured")
        
        key = self.config.encryption_key.encode()
        decrypted = bytearray()
        for i, byte in enumerate(file_content):
            decrypted.append(byte ^ key[i % len(key)])
        
        return bytes(decrypted)
    
    async def generate_thumbnail(self, filename: str, size: tuple = (200, 200)) -> bytes:
        """Generate thumbnail for image files"""
        try:
            file_content = await self.download_file(filename)
            
            # Create thumbnail
            image = Image.open(io.BytesIO(file_content))
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            return output.getvalue()
        
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            raise
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            return await self._storage_backend.get_stats()
        
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


class LocalStorage:
    """Local file system storage implementation"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize local storage"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("Local storage initialized")
    
    async def cleanup(self):
        """Cleanup local storage"""
        logger.info("Local storage cleanup completed")
    
    async def upload(self, filename: str, content: bytes):
        """Upload file to local storage"""
        file_path = self.base_path / filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
    
    async def download(self, filename: str) -> bytes:
        """Download file from local storage"""
        file_path = self.base_path / filename
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def delete(self, filename: str):
        """Delete file from local storage"""
        file_path = self.base_path / filename
        if file_path.exists():
            file_path.unlink()
    
    async def list_files(self, prefix: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files in local storage"""
        files = []
        for file_path in self.base_path.iterdir():
            if file_path.is_file() and file_path.name != "metadata":
                if prefix and not file_path.name.startswith(prefix):
                    continue
                
                metadata = await self.get_metadata(file_path.name)
                if metadata:
                    files.append(metadata)
        
        return files[offset:offset + limit]
    
    async def get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        metadata_file = self.metadata_path / f"{filename}.json"
        if metadata_file.exists():
            async with aiofiles.open(metadata_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return FileMetadata(**data)
        return None
    
    async def update_metadata(self, filename: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """Update file metadata"""
        metadata_file = self.metadata_path / f"{filename}.json"
        existing_metadata = await self.get_metadata(filename)
        
        if existing_metadata:
            existing_metadata.metadata.update(metadata)
            if tags:
                existing_metadata.tags = tags
            existing_metadata.modified_at = DateTimeHelpers.now_utc()
            
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(existing_metadata.__dict__, default=str))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_files = 0
        total_size = 0
        
        for file_path in self.base_path.iterdir():
            if file_path.is_file() and file_path.name != "metadata":
                total_files += 1
                total_size += file_path.stat().st_size
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "backend": "local"
        }


class S3Storage:
    """Amazon S3 storage implementation"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.bucket = config.s3_bucket
        self.region = config.s3_region
    
    async def initialize(self):
        """Initialize S3 storage"""
        # In real implementation, initialize S3 client
        logger.info("S3 storage initialized")
    
    async def cleanup(self):
        """Cleanup S3 storage"""
        logger.info("S3 storage cleanup completed")
    
    async def upload(self, filename: str, content: bytes):
        """Upload file to S3"""
        # In real implementation, upload to S3
        pass
    
    async def download(self, filename: str) -> bytes:
        """Download file from S3"""
        # In real implementation, download from S3
        return b""
    
    async def delete(self, filename: str):
        """Delete file from S3"""
        # In real implementation, delete from S3
        pass
    
    async def list_files(self, prefix: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files in S3"""
        # In real implementation, list S3 objects
        return []
    
    async def get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata from S3"""
        # In real implementation, get S3 object metadata
        return None
    
    async def update_metadata(self, filename: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """Update file metadata in S3"""
        # In real implementation, update S3 object metadata
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics"""
        # In real implementation, get S3 bucket statistics
        return {
            "backend": "s3",
            "bucket": self.bucket,
            "region": self.region
        }


class GCSStorage:
    """Google Cloud Storage implementation"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.bucket = config.gcs_bucket
    
    async def initialize(self):
        """Initialize GCS storage"""
        logger.info("GCS storage initialized")
    
    async def cleanup(self):
        """Cleanup GCS storage"""
        logger.info("GCS storage cleanup completed")
    
    async def upload(self, filename: str, content: bytes):
        """Upload file to GCS"""
        pass
    
    async def download(self, filename: str) -> bytes:
        """Download file from GCS"""
        return b""
    
    async def delete(self, filename: str):
        """Delete file from GCS"""
        pass
    
    async def list_files(self, prefix: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files in GCS"""
        return []
    
    async def get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata from GCS"""
        return None
    
    async def update_metadata(self, filename: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """Update file metadata in GCS"""
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get GCS storage statistics"""
        return {
            "backend": "gcs",
            "bucket": self.bucket
        }


class AzureStorage:
    """Azure Blob Storage implementation"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.container = config.azure_container
    
    async def initialize(self):
        """Initialize Azure storage"""
        logger.info("Azure storage initialized")
    
    async def cleanup(self):
        """Cleanup Azure storage"""
        logger.info("Azure storage cleanup completed")
    
    async def upload(self, filename: str, content: bytes):
        """Upload file to Azure"""
        pass
    
    async def download(self, filename: str) -> bytes:
        """Download file from Azure"""
        return b""
    
    async def delete(self, filename: str):
        """Delete file from Azure"""
        pass
    
    async def list_files(self, prefix: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files in Azure"""
        return []
    
    async def get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata from Azure"""
        return None
    
    async def update_metadata(self, filename: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """Update file metadata in Azure"""
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Azure storage statistics"""
        return {
            "backend": "azure",
            "container": self.container
        }


class MinioStorage:
    """MinIO storage implementation"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
    
    async def initialize(self):
        """Initialize MinIO storage"""
        logger.info("MinIO storage initialized")
    
    async def cleanup(self):
        """Cleanup MinIO storage"""
        logger.info("MinIO storage cleanup completed")
    
    async def upload(self, filename: str, content: bytes):
        """Upload file to MinIO"""
        pass
    
    async def download(self, filename: str) -> bytes:
        """Download file from MinIO"""
        return b""
    
    async def delete(self, filename: str):
        """Delete file from MinIO"""
        pass
    
    async def list_files(self, prefix: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files in MinIO"""
        return []
    
    async def get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata from MinIO"""
        return None
    
    async def update_metadata(self, filename: str, metadata: Dict[str, Any], tags: Optional[List[str]] = None):
        """Update file metadata in MinIO"""
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get MinIO storage statistics"""
        return {
            "backend": "minio"
        }


# Global file service
file_service = FileService()


# Utility functions
async def start_file_service():
    """Start the file service"""
    await file_service.start()


async def stop_file_service():
    """Stop the file service"""
    await file_service.stop()


async def upload_file(
    file_content: bytes,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> FileMetadata:
    """Upload file to storage"""
    return await file_service.upload_file(file_content, filename, metadata, tags)


async def download_file(filename: str) -> bytes:
    """Download file from storage"""
    return await file_service.download_file(filename)


async def delete_file(filename: str) -> bool:
    """Delete file from storage"""
    return await file_service.delete_file(filename)


async def list_files(
    prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
    file_type: Optional[FileType] = None,
    limit: int = 100,
    offset: int = 0
) -> List[FileMetadata]:
    """List files with filtering"""
    return await file_service.list_files(prefix, tags, file_type, limit, offset)


async def get_file_metadata(filename: str) -> Optional[FileMetadata]:
    """Get file metadata"""
    return await file_service.get_file_metadata(filename)


async def update_file_metadata(
    filename: str,
    metadata: Dict[str, Any],
    tags: Optional[List[str]] = None
) -> bool:
    """Update file metadata"""
    return await file_service.update_file_metadata(filename, metadata, tags)


async def generate_thumbnail(filename: str, size: tuple = (200, 200)) -> bytes:
    """Generate thumbnail for image files"""
    return await file_service.generate_thumbnail(filename, size)


async def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics"""
    return await file_service.get_storage_stats()


