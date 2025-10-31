"""
Storage Management for OpusClip Improved
=======================================

Advanced file storage system with cloud integration and local caching.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Optional, Dict, Any, List, BinaryIO, Union
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4
import aiofiles
import aiohttp
from dataclasses import dataclass
from enum import Enum
import mimetypes
import hashlib

from .schemas import get_settings
from .exceptions import StorageError, create_storage_error

logger = logging.getLogger(__name__)


class StorageType(str, Enum):
    """Storage types"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    CDN = "cdn"


class FileType(str, Enum):
    """File types"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    OTHER = "other"


@dataclass
class FileMetadata:
    """File metadata"""
    file_id: str
    filename: str
    file_type: FileType
    mime_type: str
    size: int
    checksum: str
    storage_type: StorageType
    storage_path: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None


@dataclass
class StorageConfig:
    """Storage configuration"""
    storage_type: StorageType
    base_path: str
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = None
    compression: bool = False
    encryption: bool = False
    cdn_url: Optional[str] = None


class StorageManager:
    """Advanced storage manager with multiple backends"""
    
    def __init__(self):
        self.settings = get_settings()
        self.storage_configs = {}
        self._initialize_storage_configs()
        self._ensure_directories()
    
    def _initialize_storage_configs(self):
        """Initialize storage configurations"""
        self.storage_configs = {
            StorageType.LOCAL: StorageConfig(
                storage_type=StorageType.LOCAL,
                base_path=self.settings.local_storage_path,
                max_file_size=500 * 1024 * 1024,  # 500MB
                allowed_extensions=[
                    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
                    ".mp3", ".wav", ".aac", ".flac",
                    ".jpg", ".jpeg", ".png", ".gif", ".webp",
                    ".pdf", ".doc", ".docx", ".txt"
                ]
            ),
            StorageType.S3: StorageConfig(
                storage_type=StorageType.S3,
                base_path="opus-clip",
                max_file_size=1024 * 1024 * 1024,  # 1GB
                allowed_extensions=[
                    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
                    ".mp3", ".wav", ".aac", ".flac",
                    ".jpg", ".jpeg", ".png", ".gif", ".webp"
                ],
                cdn_url=self.settings.s3_cdn_url
            ),
            StorageType.GCS: StorageConfig(
                storage_type=StorageType.GCS,
                base_path="opus-clip",
                max_file_size=1024 * 1024 * 1024,  # 1GB
                allowed_extensions=[
                    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
                    ".mp3", ".wav", ".aac", ".flac",
                    ".jpg", ".jpeg", ".png", ".gif", ".webp"
                ],
                cdn_url=self.settings.gcs_cdn_url
            )
        }
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        try:
            # Create local storage directories
            local_config = self.storage_configs[StorageType.LOCAL]
            base_path = Path(local_config.base_path)
            
            directories = [
                base_path / "videos",
                base_path / "clips",
                base_path / "thumbnails",
                base_path / "exports",
                base_path / "temp",
                base_path / "uploads"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            logger.info("Storage directories created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            raise create_storage_error("directory_creation", "storage", e)
    
    def _get_file_type(self, filename: str, mime_type: str) -> FileType:
        """Determine file type from filename and MIME type"""
        extension = Path(filename).suffix.lower()
        
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"]
        audio_extensions = [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"]
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]
        document_extensions = [".pdf", ".doc", ".docx", ".txt", ".rtf"]
        archive_extensions = [".zip", ".rar", ".7z", ".tar", ".gz"]
        
        if extension in video_extensions or mime_type.startswith("video/"):
            return FileType.VIDEO
        elif extension in audio_extensions or mime_type.startswith("audio/"):
            return FileType.AUDIO
        elif extension in image_extensions or mime_type.startswith("image/"):
            return FileType.IMAGE
        elif extension in document_extensions or mime_type.startswith("application/pdf"):
            return FileType.DOCUMENT
        elif extension in archive_extensions or mime_type.startswith("application/zip"):
            return FileType.ARCHIVE
        else:
            return FileType.OTHER
    
    def _generate_file_id(self) -> str:
        """Generate unique file ID"""
        return str(uuid4())
    
    def _generate_storage_path(self, file_id: str, filename: str, file_type: FileType) -> str:
        """Generate storage path for file"""
        extension = Path(filename).suffix.lower()
        
        # Create directory structure based on file type and date
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        
        if file_type == FileType.VIDEO:
            subdir = "videos"
        elif file_type == FileType.AUDIO:
            subdir = "audio"
        elif file_type == FileType.IMAGE:
            subdir = "images"
        else:
            subdir = "files"
        
        return f"{subdir}/{date_path}/{file_id}{extension}"
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            hash_md5 = hashlib.md5()
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(8192):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    async def store_file(
        self,
        file_data: Union[bytes, BinaryIO, str],
        filename: str,
        storage_type: StorageType = StorageType.LOCAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FileMetadata:
        """Store file in specified storage backend"""
        try:
            # Get storage configuration
            config = self.storage_configs.get(storage_type)
            if not config:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            
            # Generate file ID and path
            file_id = self._generate_file_id()
            mime_type, _ = mimetypes.guess_type(filename)
            file_type = self._get_file_type(filename, mime_type or "")
            storage_path = self._generate_storage_path(file_id, filename, file_type)
            
            # Validate file
            await self._validate_file(file_data, filename, config)
            
            # Store file based on storage type
            if storage_type == StorageType.LOCAL:
                await self._store_local(file_data, storage_path, config)
            elif storage_type == StorageType.S3:
                await self._store_s3(file_data, storage_path, config)
            elif storage_type == StorageType.GCS:
                await self._store_gcs(file_data, storage_path, config)
            else:
                raise ValueError(f"Storage type {storage_type} not implemented")
            
            # Calculate checksum
            checksum = await self._calculate_checksum_from_data(file_data)
            
            # Create file metadata
            file_metadata = FileMetadata(
                file_id=file_id,
                filename=filename,
                file_type=file_type,
                mime_type=mime_type or "application/octet-stream",
                size=await self._get_file_size(file_data),
                checksum=checksum,
                storage_type=storage_type,
                storage_path=storage_path,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            logger.info(f"File stored successfully: {file_id}")
            return file_metadata
            
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            raise create_storage_error("file_storage", filename, e)
    
    async def _validate_file(self, file_data: Union[bytes, BinaryIO, str], filename: str, config: StorageConfig):
        """Validate file before storage"""
        # Check file extension
        extension = Path(filename).suffix.lower()
        if config.allowed_extensions and extension not in config.allowed_extensions:
            raise ValueError(f"File extension {extension} not allowed")
        
        # Check file size
        file_size = await self._get_file_size(file_data)
        if file_size > config.max_file_size:
            raise ValueError(f"File size {file_size} exceeds maximum allowed size {config.max_file_size}")
    
    async def _get_file_size(self, file_data: Union[bytes, BinaryIO, str]) -> int:
        """Get file size"""
        if isinstance(file_data, bytes):
            return len(file_data)
        elif isinstance(file_data, str):
            return os.path.getsize(file_data)
        else:
            # BinaryIO
            current_pos = file_data.tell()
            file_data.seek(0, 2)  # Seek to end
            size = file_data.tell()
            file_data.seek(current_pos)  # Restore position
            return size
    
    async def _calculate_checksum_from_data(self, file_data: Union[bytes, BinaryIO, str]) -> str:
        """Calculate checksum from file data"""
        if isinstance(file_data, bytes):
            return hashlib.md5(file_data).hexdigest()
        elif isinstance(file_data, str):
            return await self._calculate_checksum(file_data)
        else:
            # BinaryIO - read all data
            current_pos = file_data.tell()
            file_data.seek(0)
            data = file_data.read()
            file_data.seek(current_pos)
            return hashlib.md5(data).hexdigest()
    
    async def _store_local(self, file_data: Union[bytes, BinaryIO, str], storage_path: str, config: StorageConfig):
        """Store file locally"""
        try:
            full_path = Path(config.base_path) / storage_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(file_data, bytes):
                async with aiofiles.open(full_path, "wb") as f:
                    await f.write(file_data)
            elif isinstance(file_data, str):
                shutil.copy2(file_data, full_path)
            else:
                # BinaryIO
                async with aiofiles.open(full_path, "wb") as f:
                    while chunk := file_data.read(8192):
                        await f.write(chunk)
            
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            raise
    
    async def _store_s3(self, file_data: Union[bytes, BinaryIO, str], storage_path: str, config: StorageConfig):
        """Store file in S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            key = f"{config.base_path}/{storage_path}"
            
            if isinstance(file_data, bytes):
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=file_data,
                    ContentType=mimetypes.guess_type(storage_path)[0] or "application/octet-stream"
                )
            elif isinstance(file_data, str):
                s3_client.upload_file(
                    file_data,
                    bucket_name,
                    key,
                    ExtraArgs={
                        'ContentType': mimetypes.guess_type(storage_path)[0] or "application/octet-stream"
                    }
                )
            else:
                # BinaryIO
                s3_client.upload_fileobj(
                    file_data,
                    bucket_name,
                    key,
                    ExtraArgs={
                        'ContentType': mimetypes.guess_type(storage_path)[0] or "application/octet-stream"
                    }
                )
            
        except Exception as e:
            logger.error(f"S3 storage failed: {e}")
            raise
    
    async def _store_gcs(self, file_data: Union[bytes, BinaryIO, str], storage_path: str, config: StorageConfig):
        """Store file in Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            blob = bucket.blob(f"{config.base_path}/{storage_path}")
            
            if isinstance(file_data, bytes):
                blob.upload_from_string(file_data)
            elif isinstance(file_data, str):
                blob.upload_from_filename(file_data)
            else:
                # BinaryIO
                blob.upload_from_file(file_data)
            
        except Exception as e:
            logger.error(f"GCS storage failed: {e}")
            raise
    
    async def retrieve_file(self, file_metadata: FileMetadata) -> bytes:
        """Retrieve file from storage"""
        try:
            if file_metadata.storage_type == StorageType.LOCAL:
                return await self._retrieve_local(file_metadata)
            elif file_metadata.storage_type == StorageType.S3:
                return await self._retrieve_s3(file_metadata)
            elif file_metadata.storage_type == StorageType.GCS:
                return await self._retrieve_gcs(file_metadata)
            else:
                raise ValueError(f"Storage type {file_metadata.storage_type} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to retrieve file: {e}")
            raise create_storage_error("file_retrieval", file_metadata.file_id, e)
    
    async def _retrieve_local(self, file_metadata: FileMetadata) -> bytes:
        """Retrieve file from local storage"""
        try:
            config = self.storage_configs[StorageType.LOCAL]
            full_path = Path(config.base_path) / file_metadata.storage_path
            
            async with aiofiles.open(full_path, "rb") as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Local file retrieval failed: {e}")
            raise
    
    async def _retrieve_s3(self, file_metadata: FileMetadata) -> bytes:
        """Retrieve file from S3"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            key = f"{self.storage_configs[StorageType.S3].base_path}/{file_metadata.storage_path}"
            
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"S3 file retrieval failed: {e}")
            raise
    
    async def _retrieve_gcs(self, file_metadata: FileMetadata) -> bytes:
        """Retrieve file from Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            blob = bucket.blob(f"{self.storage_configs[StorageType.GCS].base_path}/{file_metadata.storage_path}")
            
            return blob.download_as_bytes()
            
        except Exception as e:
            logger.error(f"GCS file retrieval failed: {e}")
            raise
    
    async def delete_file(self, file_metadata: FileMetadata) -> bool:
        """Delete file from storage"""
        try:
            if file_metadata.storage_type == StorageType.LOCAL:
                return await self._delete_local(file_metadata)
            elif file_metadata.storage_type == StorageType.S3:
                return await self._delete_s3(file_metadata)
            elif file_metadata.storage_type == StorageType.GCS:
                return await self._delete_gcs(file_metadata)
            else:
                raise ValueError(f"Storage type {file_metadata.storage_type} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    async def _delete_local(self, file_metadata: FileMetadata) -> bool:
        """Delete file from local storage"""
        try:
            config = self.storage_configs[StorageType.LOCAL]
            full_path = Path(config.base_path) / file_metadata.storage_path
            
            if full_path.exists():
                full_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Local file deletion failed: {e}")
            return False
    
    async def _delete_s3(self, file_metadata: FileMetadata) -> bool:
        """Delete file from S3"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            key = f"{self.storage_configs[StorageType.S3].base_path}/{file_metadata.storage_path}"
            
            s3_client.delete_object(Bucket=bucket_name, Key=key)
            return True
            
        except Exception as e:
            logger.error(f"S3 file deletion failed: {e}")
            return False
    
    async def _delete_gcs(self, file_metadata: FileMetadata) -> bool:
        """Delete file from Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            blob = bucket.blob(f"{self.storage_configs[StorageType.GCS].base_path}/{file_metadata.storage_path}")
            
            blob.delete()
            return True
            
        except Exception as e:
            logger.error(f"GCS file deletion failed: {e}")
            return False
    
    async def get_file_url(self, file_metadata: FileMetadata, expires_in: int = 3600) -> str:
        """Get public URL for file"""
        try:
            if file_metadata.storage_type == StorageType.LOCAL:
                # For local storage, return a relative URL
                return f"/files/{file_metadata.file_id}"
            elif file_metadata.storage_type == StorageType.S3:
                return await self._get_s3_url(file_metadata, expires_in)
            elif file_metadata.storage_type == StorageType.GCS:
                return await self._get_gcs_url(file_metadata, expires_in)
            else:
                raise ValueError(f"Storage type {file_metadata.storage_type} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to get file URL: {e}")
            raise create_storage_error("url_generation", file_metadata.file_id, e)
    
    async def _get_s3_url(self, file_metadata: FileMetadata, expires_in: int) -> str:
        """Get S3 presigned URL"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            key = f"{self.storage_configs[StorageType.S3].base_path}/{file_metadata.storage_path}"
            
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            
            return url
            
        except Exception as e:
            logger.error(f"S3 URL generation failed: {e}")
            raise
    
    async def _get_gcs_url(self, file_metadata: FileMetadata, expires_in: int) -> str:
        """Get GCS signed URL"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            blob = bucket.blob(f"{self.storage_configs[StorageType.GCS].base_path}/{file_metadata.storage_path}")
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.utcnow() + timedelta(seconds=expires_in),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            logger.error(f"GCS URL generation failed: {e}")
            raise
    
    async def list_files(
        self,
        storage_type: StorageType = StorageType.LOCAL,
        prefix: str = "",
        limit: int = 100
    ) -> List[FileMetadata]:
        """List files in storage"""
        try:
            if storage_type == StorageType.LOCAL:
                return await self._list_local_files(prefix, limit)
            elif storage_type == StorageType.S3:
                return await self._list_s3_files(prefix, limit)
            elif storage_type == StorageType.GCS:
                return await self._list_gcs_files(prefix, limit)
            else:
                raise ValueError(f"Storage type {storage_type} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise create_storage_error("file_listing", prefix, e)
    
    async def _list_local_files(self, prefix: str, limit: int) -> List[FileMetadata]:
        """List files in local storage"""
        try:
            config = self.storage_configs[StorageType.LOCAL]
            base_path = Path(config.base_path)
            
            files = []
            for file_path in base_path.rglob(f"{prefix}*"):
                if file_path.is_file():
                    file_metadata = FileMetadata(
                        file_id=file_path.stem,
                        filename=file_path.name,
                        file_type=self._get_file_type(file_path.name, ""),
                        mime_type=mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
                        size=file_path.stat().st_size,
                        checksum="",
                        storage_type=StorageType.LOCAL,
                        storage_path=str(file_path.relative_to(base_path)),
                        created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                        updated_at=datetime.fromtimestamp(file_path.stat().st_mtime)
                    )
                    files.append(file_metadata)
                    
                    if len(files) >= limit:
                        break
            
            return files
            
        except Exception as e:
            logger.error(f"Local file listing failed: {e}")
            raise
    
    async def _list_s3_files(self, prefix: str, limit: int) -> List[FileMetadata]:
        """List files in S3"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            config = self.storage_configs[StorageType.S3]
            full_prefix = f"{config.base_path}/{prefix}"
            
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=full_prefix,
                MaxKeys=limit
            )
            
            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                filename = key.split('/')[-1]
                storage_path = key[len(config.base_path) + 1:]
                
                file_metadata = FileMetadata(
                    file_id=obj['ETag'].strip('"'),
                    filename=filename,
                    file_type=self._get_file_type(filename, ""),
                    mime_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    size=obj['Size'],
                    checksum=obj['ETag'].strip('"'),
                    storage_type=StorageType.S3,
                    storage_path=storage_path,
                    created_at=obj['LastModified'],
                    updated_at=obj['LastModified']
                )
                files.append(file_metadata)
            
            return files
            
        except Exception as e:
            logger.error(f"S3 file listing failed: {e}")
            raise
    
    async def _list_gcs_files(self, prefix: str, limit: int) -> List[FileMetadata]:
        """List files in Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            config = self.storage_configs[StorageType.GCS]
            full_prefix = f"{config.base_path}/{prefix}"
            
            blobs = bucket.list_blobs(prefix=full_prefix, max_results=limit)
            
            files = []
            for blob in blobs:
                filename = blob.name.split('/')[-1]
                storage_path = blob.name[len(config.base_path) + 1:]
                
                file_metadata = FileMetadata(
                    file_id=blob.etag,
                    filename=filename,
                    file_type=self._get_file_type(filename, ""),
                    mime_type=blob.content_type or "application/octet-stream",
                    size=blob.size,
                    checksum=blob.etag,
                    storage_type=StorageType.GCS,
                    storage_path=storage_path,
                    created_at=blob.time_created,
                    updated_at=blob.updated
                )
                files.append(file_metadata)
            
            return files
            
        except Exception as e:
            logger.error(f"GCS file listing failed: {e}")
            raise
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified age"""
        try:
            config = self.storage_configs[StorageType.LOCAL]
            temp_path = Path(config.base_path) / "temp"
            
            if not temp_path.exists():
                return 0
            
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                "local": await self._get_local_stats(),
                "s3": await self._get_s3_stats(),
                "gcs": await self._get_gcs_stats()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def _get_local_stats(self) -> Dict[str, Any]:
        """Get local storage statistics"""
        try:
            config = self.storage_configs[StorageType.LOCAL]
            base_path = Path(config.base_path)
            
            total_size = 0
            file_count = 0
            
            for file_path in base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "total_size": total_size,
                "file_count": file_count,
                "available_space": shutil.disk_usage(base_path).free
            }
            
        except Exception as e:
            logger.error(f"Failed to get local storage stats: {e}")
            return {"error": str(e)}
    
    async def _get_s3_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            bucket_name = self.settings.s3_bucket_name
            config = self.storage_configs[StorageType.S3]
            prefix = f"{config.base_path}/"
            
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            file_count = len(response.get('Contents', []))
            
            return {
                "total_size": total_size,
                "file_count": file_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get S3 storage stats: {e}")
            return {"error": str(e)}
    
    async def _get_gcs_stats(self) -> Dict[str, Any]:
        """Get Google Cloud Storage statistics"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.settings.gcs_bucket_name)
            config = self.storage_configs[StorageType.GCS]
            prefix = f"{config.base_path}/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            
            total_size = 0
            file_count = 0
            
            for blob in blobs:
                total_size += blob.size
                file_count += 1
            
            return {
                "total_size": total_size,
                "file_count": file_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCS storage stats: {e}")
            return {"error": str(e)}


# Global storage manager
storage_manager = StorageManager()





























