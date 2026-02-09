from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import uuid
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import aiofiles
import mimetypes
from core.config import get_config
from core.exceptions import FileError, ValidationError, handle_async_exception
from core.types import FileInfo, ContentType
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
File Service for OS Content UGC Video Generator
Handles file operations and validation
"""



logger = structlog.get_logger("os_content.file_service")

class FileService:
    """File handling service"""
    
    def __init__(self) -> Any:
        self.config = get_config()
    
    @handle_async_exception
    async async def save_uploaded_file(self, file_path: str, content: bytes) -> str:
        """Save uploaded file to storage"""
        
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            extension = Path(file_path).suffix
            filename = f"{file_id}{extension}"
            
            # Create full path
            full_path = Path(self.config.storage.upload_dir) / filename
            
            # Save file
            async with aiofiles.open(full_path, "wb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            logger.info(f"File saved: {full_path}")
            return str(full_path)
            
        except Exception as e:
            raise FileError(f"Failed to save file: {e}", file_path=file_path, operation="save")
    
    @handle_async_exception
    async async def process_uploaded_files(self, file_paths: List[str], content_type: str) -> List[str]:
        """Process uploaded files and return valid paths"""
        
        processed_paths = []
        
        for file_path in file_paths:
            try:
                # Validate file
                file_info = await self.get_file_info(file_path)
                await self.validate_file(file_info, content_type)
                
                processed_paths.append(file_path)
                
            except Exception as e:
                logger.warning(f"File validation failed: {file_path} - {e}")
                continue
        
        return processed_paths
    
    @handle_async_exception
    async def get_file_info(self, file_path: str) -> FileInfo:
        """Get file information"""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileError(f"File not found: {file_path}", file_path=file_path, operation="read")
        
        # Get file stats
        stat = path.stat()
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(str(path))
        if not content_type:
            content_type = "application/octet-stream"
        
        # Calculate checksum
        checksum = await self.calculate_checksum(file_path)
        
        return FileInfo(
            path=file_path,
            size=stat.st_size,
            content_type=content_type,
            extension=path.suffix.lower(),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            checksum=checksum
        )
    
    @handle_async_exception
    async def validate_file(self, file_info: FileInfo, expected_type: str) -> bool:
        """Validate file based on type and size"""
        
        # Check file size
        if file_info.size > self.config.storage.max_file_size:
            raise ValidationError(
                f"File too large: {file_info.size} bytes (max: {self.config.storage.max_file_size})",
                field="file_size",
                value=file_info.size
            )
        
        # Check file extension
        if file_info.extension not in self.config.storage.allowed_extensions:
            raise ValidationError(
                f"File extension not allowed: {file_info.extension}",
                field="file_extension",
                value=file_info.extension
            )
        
        # Validate content type
        if expected_type == "image":
            if not file_info.content_type.startswith("image/"):
                raise ValidationError(
                    f"Invalid content type for image: {file_info.content_type}",
                    field="content_type",
                    value=file_info.content_type
                )
        elif expected_type == "video":
            if not file_info.content_type.startswith("video/"):
                raise ValidationError(
                    f"Invalid content type for video: {file_info.content_type}",
                    field="content_type",
                    value=file_info.content_type
                )
        
        return True
    
    @handle_async_exception
    async def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            while chunk := await f.read(8192):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    @handle_async_exception
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage"""
        
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
            
        except Exception as e:
            raise FileError(f"Failed to delete file: {e}", file_path=file_path, operation="delete")
    
    @handle_async_exception
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified age"""
        
        try:
            upload_dir = Path(self.config.storage.upload_dir)
            current_time = datetime.now()
            deleted_count = 0
            
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    # Check file age
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0
    
    @handle_async_exception
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        try:
            upload_dir = Path(self.config.storage.upload_dir)
            
            total_files = 0
            total_size = 0
            file_types = {}
            
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size": total_size,
                "file_types": file_types,
                "upload_dir": str(upload_dir),
                "max_file_size": self.config.storage.max_file_size,
                "allowed_extensions": self.config.storage.allowed_extensions
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    @handle_async_exception
    async def validate_file_path(self, file_path: str) -> bool:
        """Validate file path is within allowed directory"""
        
        try:
            path = Path(file_path).resolve()
            upload_dir = Path(self.config.storage.upload_dir).resolve()
            
            # Check if path is within upload directory
            return upload_dir in path.parents or path == upload_dir
            
        except Exception:
            return False

# Global file service instance
file_service = FileService() 