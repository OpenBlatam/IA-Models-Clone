"""
Gamma App - File Service
Advanced file management and storage service
"""

import os
import hashlib
import mimetypes
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum
import uuid
import shutil
import tempfile
from datetime import datetime, timedelta
import logging
import boto3
from botocore.exceptions import ClientError
import magic

logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Storage types"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

class FileType(Enum):
    """File types"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    CODE = "code"
    DATA = "data"
    OTHER = "other"

@dataclass
class FileMetadata:
    """File metadata"""
    id: str
    filename: str
    original_filename: str
    file_type: FileType
    mime_type: str
    size: int
    hash: str
    storage_type: StorageType
    storage_path: str
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class UploadConfig:
    """Upload configuration"""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = None
    allowed_mime_types: List[str] = None
    generate_thumbnail: bool = True
    compress_images: bool = True
    virus_scan: bool = True
    storage_type: StorageType = StorageType.LOCAL
    expires_in_days: Optional[int] = None

class FileService:
    """Advanced file management service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_config = config.get("storage", {})
        self.upload_dir = Path(self.storage_config.get("upload_dir", "./uploads"))
        self.temp_dir = Path(self.storage_config.get("temp_dir", "./temp"))
        self.thumbnail_dir = Path(self.storage_config.get("thumbnail_dir", "./thumbnails"))
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage clients
        self._init_storage_clients()
        
        # File type mappings
        self._init_file_type_mappings()
    
    def _init_storage_clients(self):
        """Initialize storage clients"""
        self.s3_client = None
        self.gcs_client = None
        self.azure_client = None
        
        # S3 client
        if self.storage_config.get("s3", {}).get("enabled"):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.storage_config["s3"]["access_key"],
                aws_secret_access_key=self.storage_config["s3"]["secret_key"],
                region_name=self.storage_config["s3"]["region"]
            )
        
        # GCS client (would initialize here)
        if self.storage_config.get("gcs", {}).get("enabled"):
            try:
                from google.cloud import storage
                self.gcs_client = storage.Client()
                self.gcs_bucket = self.gcs_client.bucket(
                    self.storage_config["gcs"]["bucket_name"]
                )
                logger.info("GCS client initialized")
            except ImportError:
                logger.warning("Google Cloud Storage not available")
            except Exception as e:
                logger.error(f"Error initializing GCS client: {e}")
        
        # Azure client (would initialize here)
        if self.storage_config.get("azure", {}).get("enabled"):
            try:
                from azure.storage.blob import BlobServiceClient
                self.azure_client = BlobServiceClient.from_connection_string(
                    self.storage_config["azure"]["connection_string"]
                )
                self.azure_container = self.azure_client.get_container_client(
                    self.storage_config["azure"]["container_name"]
                )
                logger.info("Azure client initialized")
            except ImportError:
                logger.warning("Azure Storage not available")
            except Exception as e:
                logger.error(f"Error initializing Azure client: {e}")
    
    def _init_file_type_mappings(self):
        """Initialize file type mappings"""
        self.file_type_mappings = {
            # Images
            'image/jpeg': FileType.IMAGE,
            'image/png': FileType.IMAGE,
            'image/gif': FileType.IMAGE,
            'image/webp': FileType.IMAGE,
            'image/svg+xml': FileType.IMAGE,
            'image/bmp': FileType.IMAGE,
            'image/tiff': FileType.IMAGE,
            
            # Videos
            'video/mp4': FileType.VIDEO,
            'video/avi': FileType.VIDEO,
            'video/mov': FileType.VIDEO,
            'video/wmv': FileType.VIDEO,
            'video/flv': FileType.VIDEO,
            'video/webm': FileType.VIDEO,
            'video/mkv': FileType.VIDEO,
            
            # Audio
            'audio/mp3': FileType.AUDIO,
            'audio/wav': FileType.AUDIO,
            'audio/flac': FileType.AUDIO,
            'audio/aac': FileType.AUDIO,
            'audio/ogg': FileType.AUDIO,
            'audio/m4a': FileType.AUDIO,
            
            # Documents
            'application/pdf': FileType.DOCUMENT,
            'application/msword': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCUMENT,
            'application/vnd.ms-excel': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileType.DOCUMENT,
            'application/vnd.ms-powerpoint': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileType.DOCUMENT,
            'text/plain': FileType.DOCUMENT,
            'text/html': FileType.DOCUMENT,
            'text/markdown': FileType.DOCUMENT,
            'application/json': FileType.DOCUMENT,
            'application/xml': FileType.DOCUMENT,
            
            # Archives
            'application/zip': FileType.ARCHIVE,
            'application/x-rar-compressed': FileType.ARCHIVE,
            'application/x-7z-compressed': FileType.ARCHIVE,
            'application/gzip': FileType.ARCHIVE,
            'application/x-tar': FileType.ARCHIVE,
            
            # Code
            'text/x-python': FileType.CODE,
            'text/javascript': FileType.CODE,
            'text/css': FileType.CODE,
            'text/x-java': FileType.CODE,
            'text/x-c': FileType.CODE,
            'text/x-c++': FileType.CODE,
            'text/x-csharp': FileType.CODE,
            'text/x-php': FileType.CODE,
            'text/x-ruby': FileType.CODE,
            'text/x-go': FileType.CODE,
            
            # Data
            'text/csv': FileType.DATA,
            'application/x-sql': FileType.DATA,
            'application/x-parquet': FileType.DATA,
        }
    
    async def upload_file(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        config: UploadConfig = None
    ) -> FileMetadata:
        """Upload a file"""
        if config is None:
            config = UploadConfig()
        
        try:
            # Read file data
            if isinstance(file_data, bytes):
                content = file_data
            else:
                content = await file_data.read()
            
            # Validate file
            validation_result = await self._validate_file(content, filename, config)
            if not validation_result["valid"]:
                raise ValueError(f"File validation failed: {validation_result['error']}")
            
            # Generate file metadata
            file_id = str(uuid.uuid4())
            file_hash = hashlib.sha256(content).hexdigest()
            mime_type = magic.from_buffer(content, mime=True)
            file_type = self.file_type_mappings.get(mime_type, FileType.OTHER)
            
            # Generate storage path
            storage_path = self._generate_storage_path(file_id, filename)
            
            # Upload to storage
            url = await self._upload_to_storage(content, storage_path, config.storage_type)
            
            # Create metadata
            metadata = FileMetadata(
                id=file_id,
                filename=filename,
                original_filename=filename,
                file_type=file_type,
                mime_type=mime_type,
                size=len(content),
                hash=file_hash,
                storage_type=config.storage_type,
                storage_path=storage_path,
                url=url,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=config.expires_in_days) if config.expires_in_days else None,
                metadata={}
            )
            
            # Generate thumbnail if needed
            if config.generate_thumbnail and file_type == FileType.IMAGE:
                thumbnail_url = await self._generate_thumbnail(content, file_id)
                metadata.thumbnail_url = thumbnail_url
            
            # Store metadata
            await self._store_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download a file by ID"""
        try:
            metadata = await self._get_metadata(file_id)
            if not metadata:
                return None
            
            return await self._download_from_storage(metadata.storage_path, metadata.storage_type)
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
    
    async def get_file_url(self, file_id: str, expires_in: int = 3600) -> Optional[str]:
        """Get a signed URL for file access"""
        try:
            metadata = await self._get_metadata(file_id)
            if not metadata:
                return None
            
            if metadata.storage_type == StorageType.LOCAL:
                return metadata.url
            elif metadata.storage_type == StorageType.S3:
                return self._generate_s3_presigned_url(metadata.storage_path, expires_in)
            else:
                return metadata.url
                
        except Exception as e:
            logger.error(f"Error getting file URL: {e}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        try:
            metadata = await self._get_metadata(file_id)
            if not metadata:
                return False
            
            # Delete from storage
            await self._delete_from_storage(metadata.storage_path, metadata.storage_type)
            
            # Delete metadata
            await self._delete_metadata(file_id)
            
            # Delete thumbnail if exists
            if metadata.thumbnail_url:
                await self._delete_thumbnail(file_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def list_files(
        self,
        file_type: Optional[FileType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[FileMetadata]:
        """List files with optional filtering"""
        try:
            # This would query the database for file metadata
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def search_files(self, query: str, limit: int = 100) -> List[FileMetadata]:
        """Search files by filename or metadata"""
        try:
            # This would implement file search
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []
    
    async def get_file_stats(self) -> Dict[str, Any]:
        """Get file storage statistics"""
        try:
            stats = {
                "total_files": 0,
                "total_size": 0,
                "by_type": {},
                "by_storage": {},
                "recent_uploads": 0
            }
            
            # This would query the database for statistics
            # For now, return empty stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting file stats: {e}")
            return {}
    
    async def cleanup_expired_files(self) -> int:
        """Clean up expired files"""
        try:
            # This would find and delete expired files
            # For now, return 0
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up expired files: {e}")
            return 0
    
    async def _validate_file(
        self,
        content: bytes,
        filename: str,
        config: UploadConfig
    ) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check file size
            if len(content) > config.max_file_size:
                return {"valid": False, "error": "File size exceeds limit"}
            
            # Check file extension
            if config.allowed_extensions:
                file_ext = Path(filename).suffix.lower()
                if file_ext not in config.allowed_extensions:
                    return {"valid": False, "error": "File extension not allowed"}
            
            # Check MIME type
            if config.allowed_mime_types:
                mime_type = magic.from_buffer(content, mime=True)
                if mime_type not in config.allowed_mime_types:
                    return {"valid": False, "error": "File type not allowed"}
            
            # Virus scan (if enabled)
            if config.virus_scan:
                # Basic virus scanning implementation
                try:
                    # Check file extension against known malicious extensions
                    malicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.vbs', '.js']
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in malicious_extensions:
                        return {"valid": False, "reason": "Potentially malicious file extension"}
                    
                    # Check file size (basic heuristic)
                    file_size = Path(file_path).stat().st_size
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        return {"valid": False, "reason": "File too large for scanning"}
                    
                    # In a real implementation, you would integrate with ClamAV or similar
                    logger.info(f"Basic virus scan passed for {file_path}")
                except Exception as e:
                    logger.warning(f"Virus scan failed: {e}")
                    return {"valid": False, "reason": "Virus scan failed"}
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {"valid": False, "error": "Validation error"}
    
    def _generate_storage_path(self, file_id: str, filename: str) -> str:
        """Generate storage path for file"""
        # Create directory structure based on file ID
        dir1 = file_id[:2]
        dir2 = file_id[2:4]
        
        # Get file extension
        ext = Path(filename).suffix
        
        return f"{dir1}/{dir2}/{file_id}{ext}"
    
    async def _upload_to_storage(
        self,
        content: bytes,
        storage_path: str,
        storage_type: StorageType
    ) -> str:
        """Upload content to storage"""
        if storage_type == StorageType.LOCAL:
            return await self._upload_to_local(content, storage_path)
        elif storage_type == StorageType.S3:
            return await self._upload_to_s3(content, storage_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    async def _upload_to_local(self, content: bytes, storage_path: str) -> str:
        """Upload to local storage"""
        file_path = self.upload_dir / storage_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return f"/uploads/{storage_path}"
    
    async def _upload_to_s3(self, content: bytes, storage_path: str) -> str:
        """Upload to S3"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        bucket = self.storage_config["s3"]["bucket"]
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=storage_path,
                Body=content,
                ContentType=mimetypes.guess_type(storage_path)[0] or 'application/octet-stream'
            )
            
            return f"https://{bucket}.s3.amazonaws.com/{storage_path}"
            
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    async def _download_from_storage(
        self,
        storage_path: str,
        storage_type: StorageType
    ) -> bytes:
        """Download content from storage"""
        if storage_type == StorageType.LOCAL:
            return await self._download_from_local(storage_path)
        elif storage_type == StorageType.S3:
            return await self._download_from_s3(storage_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    async def _download_from_local(self, storage_path: str) -> bytes:
        """Download from local storage"""
        file_path = self.upload_dir / storage_path
        
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def _download_from_s3(self, storage_path: str) -> bytes:
        """Download from S3"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        bucket = self.storage_config["s3"]["bucket"]
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=storage_path)
            return response['Body'].read()
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
    
    async def _delete_from_storage(
        self,
        storage_path: str,
        storage_type: StorageType
    ):
        """Delete content from storage"""
        if storage_type == StorageType.LOCAL:
            await self._delete_from_local(storage_path)
        elif storage_type == StorageType.S3:
            await self._delete_from_s3(storage_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    async def _delete_from_local(self, storage_path: str):
        """Delete from local storage"""
        file_path = self.upload_dir / storage_path
        if file_path.exists():
            file_path.unlink()
    
    async def _delete_from_s3(self, storage_path: str):
        """Delete from S3"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        bucket = self.storage_config["s3"]["bucket"]
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=storage_path)
            
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            raise
    
    def _generate_s3_presigned_url(self, storage_path: str, expires_in: int) -> str:
        """Generate S3 presigned URL"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        bucket = self.storage_config["s3"]["bucket"]
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': storage_path},
                ExpiresIn=expires_in
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise
    
    async def _generate_thumbnail(self, content: bytes, file_id: str) -> str:
        """Generate thumbnail for image"""
        try:
            from PIL import Image
            import io
            
            # Open image
            image = Image.open(io.BytesIO(content))
            
            # Create thumbnail
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = self.thumbnail_dir / f"{file_id}_thumb.jpg"
            image.save(thumbnail_path, "JPEG", quality=85)
            
            return f"/thumbnails/{file_id}_thumb.jpg"
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return None
    
    async def _delete_thumbnail(self, file_id: str):
        """Delete thumbnail"""
        thumbnail_path = self.thumbnail_dir / f"{file_id}_thumb.jpg"
        if thumbnail_path.exists():
            thumbnail_path.unlink()
    
    async def _store_metadata(self, metadata: FileMetadata):
        """Store file metadata"""
        # This would store metadata in database
        # For now, just log
        logger.info(f"Storing metadata for file {metadata.id}")
    
    async def _get_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        # This would retrieve metadata from database
        # For now, return None
        return None
    
    async def _delete_metadata(self, file_id: str):
        """Delete file metadata"""
        # This would delete metadata from database
        # For now, just log
        logger.info(f"Deleting metadata for file {file_id}")
    
    async def copy_file(self, source_file_id: str, new_filename: str) -> Optional[FileMetadata]:
        """Copy a file"""
        try:
            # Get source file
            source_metadata = await self._get_metadata(source_file_id)
            if not source_metadata:
                return None
            
            # Download source content
            content = await self.download_file(source_file_id)
            if not content:
                return None
            
            # Upload as new file
            new_metadata = await self.upload_file(content, new_filename)
            
            return new_metadata
            
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return None
    
    async def move_file(self, file_id: str, new_storage_type: StorageType) -> bool:
        """Move file to different storage type"""
        try:
            # Get file metadata
            metadata = await self._get_metadata(file_id)
            if not metadata:
                return False
            
            # Download from current storage
            content = await self.download_file(file_id)
            if not content:
                return False
            
            # Upload to new storage
            new_url = await self._upload_to_storage(content, metadata.storage_path, new_storage_type)
            
            # Delete from old storage
            await self._delete_from_storage(metadata.storage_path, metadata.storage_type)
            
            # Update metadata
            metadata.storage_type = new_storage_type
            metadata.url = new_url
            metadata.updated_at = datetime.now()
            
            await self._store_metadata(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return False




