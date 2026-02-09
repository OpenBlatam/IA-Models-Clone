from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Optional, BinaryIO, Dict, Any, List
import os
import uuid
import hashlib
import aiofiles
import aiofiles.os
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import mimetypes
from fastapi import UploadFile, HTTPException
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
import json
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from typing import Any, List, Dict, Optional
import logging
"""
Optimized storage service for handling file uploads and retrievals with async operations.
"""


logger = setup_logger()

class OptimizedStorageService:
    """Optimized service for handling file storage operations with caching and async operations."""
    
    def __init__(self) -> Any:
        """Initialize the service with optimized settings."""
        self.base_path = Path(settings.storage_path)
        self.cache_ttl = settings.image_cache_ttl
        self.max_file_size = settings.max_file_size
        self.allowed_extensions = settings.allowed_file_types
        self._redis_client = None
        self._ensure_storage_path()
    
    def _ensure_storage_path(self) -> Any:
        """Ensure the storage path exists with proper permissions."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for better organization
            (self.base_path / "images").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            (self.base_path / "processed").mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            raise
    
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client for caching."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    def _get_file_path(self, filename: str, subdirectory: str = "") -> Path:
        """
        Get the full path for a file with optional subdirectory.
        
        Args:
            filename: The filename to get the path for
            subdirectory: Optional subdirectory within storage
            
        Returns:
            The full path for the file
        """
        if subdirectory:
            return self.base_path / subdirectory / filename
        return self.base_path / filename
    
    def _generate_filename(self, original_filename: str, prefix: str = "") -> str:
        """
        Generate a unique filename with optional prefix.
        
        Args:
            original_filename: The original filename
            prefix: Optional prefix for the filename
            
        Returns:
            A unique filename
        """
        ext = Path(original_filename).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        prefix_part = f"{prefix}_" if prefix else ""
        return f"{prefix_part}{timestamp}_{unique_id}{ext}"
    
    def _validate_file_type(self, filename: str) -> bool:
        """Validate if file type is allowed."""
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in self.allowed_extensions
    
    def _validate_file_size(self, file_size: int) -> bool:
        """Validate if file size is within limits."""
        return file_size <= self.max_file_size
    
    async def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for storage operations."""
        key_data = json.dumps(kwargs, sort_keys=True)
        return f"storage:{operation}:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _get_cached_file_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached file information."""
        try:
            redis = await self.redis_client
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _set_cached_file_info(self, cache_key: str, file_info: Dict[str, Any], ttl: int = None):
        """Set file information in cache."""
        try:
            redis = await self.redis_client
            ttl = ttl or self.cache_ttl
            await redis.setex(cache_key, ttl, json.dumps(file_info))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    @asynccontextmanager
    async def _temp_file_context(self, filename: str):
        """Context manager for temporary file operations."""
        temp_path = self._get_file_path(filename, "temp")
        try:
            yield temp_path
        finally:
            # Clean up temp file
            try:
                if await aiofiles.os.path.exists(temp_path):
                    await aiofiles.os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
    
    async def save_file(
        self,
        file: BinaryIO,
        original_filename: str,
        subdirectory: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a file to storage with optimized async operations.
        
        Args:
            file: The file to save
            original_filename: The original filename
            subdirectory: Optional subdirectory within storage
            metadata: Optional metadata about the file
            
        Returns:
            The filename of the saved file
        """
        try:
            # Validate file type
            if not self._validate_file_type(original_filename):
                raise ValueError(f"File type not allowed: {original_filename}")
            
            # Generate unique filename
            filename = self._generate_filename(original_filename)
            file_path = self._get_file_path(filename, subdirectory)
            
            # Ensure subdirectory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file with chunked writing for large files
            async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                chunk_size = settings.chunk_size
                total_size = 0
                
                while True:
                    chunk = await file.read(chunk_size)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    if not chunk:
                        break
                    
                    # Check file size limit
                    total_size += len(chunk)
                    if not self._validate_file_size(total_size):
                        raise ValueError(f"File too large: {total_size} bytes")
                    
                    await f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Store file metadata in cache
            file_info = {
                "filename": filename,
                "original_filename": original_filename,
                "file_size": total_size,
                "mime_type": mimetypes.guess_type(original_filename)[0],
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            cache_key = await self._get_cache_key("file_info", filename=filename)
            await self._set_cached_file_info(cache_key, file_info)
            
            logger.info(f"File saved successfully: {filename} ({total_size} bytes)")
            return filename
            
        except Exception as e:
            logger.exception("Error saving file")
            raise
    
    async async def save_upload_file(
        self,
        upload_file: UploadFile,
        subdirectory: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save an uploaded file to storage with validation.
        
        Args:
            upload_file: The uploaded file
            subdirectory: Optional subdirectory within storage
            metadata: Optional metadata about the file
            
        Returns:
            The filename of the saved file
        """
        try:
            # Validate file type
            if not self._validate_file_type(upload_file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type not allowed: {upload_file.filename}"
                )
            
            # Validate file size
            if upload_file.size and not self._validate_file_size(upload_file.size):
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {upload_file.size} bytes"
                )
            
            # Generate unique filename
            filename = self._generate_filename(upload_file.filename)
            file_path = self._get_file_path(filename, subdirectory)
            
            # Ensure subdirectory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file with progress tracking
            total_size = 0
            async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                chunk_size = settings.chunk_size
                
                while True:
                    chunk = await upload_file.read(chunk_size)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    if not chunk:
                        break
                    
                    total_size += len(chunk)
                    if not self._validate_file_size(total_size):
                        # Clean up partial file
                        await aiofiles.os.remove(file_path)
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large: {total_size} bytes"
                        )
                    
                    await f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Store file metadata
            file_info = {
                "filename": filename,
                "original_filename": upload_file.filename,
                "file_size": total_size,
                "mime_type": upload_file.content_type,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            cache_key = await self._get_cache_key("file_info", filename=filename)
            await self._set_cached_file_info(cache_key, file_info)
            
            logger.info(f"Upload file saved: {filename} ({total_size} bytes)")
            return filename
            
        except Exception as e:
            logger.exception("Error saving uploaded file")
            raise
    
    async def get_file(
        self,
        filename: str,
        subdirectory: str = ""
    ) -> Optional[bytes]:
        """
        Get a file from storage with caching.
        
        Args:
            filename: The filename to get
            subdirectory: Optional subdirectory within storage
            
        Returns:
            The file contents if found, None otherwise
        """
        try:
            file_path = self._get_file_path(filename, subdirectory)
            
            if not await aiofiles.os.path.exists(file_path):
                return None
            
            # Check cache for file info
            cache_key = await self._get_cache_key("file_info", filename=filename)
            cached_info = await self._get_cached_file_info(cache_key)
            
            async with aiofiles.open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Update cache with access time
            if cached_info:
                cached_info["last_accessed"] = datetime.utcnow().isoformat()
                await self._set_cached_file_info(cache_key, cached_info)
            
            return content
            
        except Exception as e:
            logger.exception("Error getting file")
            raise
    
    async def get_file_info(
        self,
        filename: str,
        subdirectory: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get file information with caching.
        
        Args:
            filename: The filename to get info for
            subdirectory: Optional subdirectory within storage
            
        Returns:
            File information if found, None otherwise
        """
        try:
            # Check cache first
            cache_key = await self._get_cache_key("file_info", filename=filename)
            cached_info = await self._get_cached_file_info(cache_key)
            
            if cached_info:
                return cached_info
            
            file_path = self._get_file_path(filename, subdirectory)
            
            if not await aiofiles.os.path.exists(file_path):
                return None
            
            # Get file stats
            stat = await aiofiles.os.stat(file_path)
            
            file_info = {
                "filename": filename,
                "file_size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "mime_type": mimetypes.guess_type(filename)[0]
            }
            
            # Cache the info
            await self._set_cached_file_info(cache_key, file_info)
            
            return file_info
            
        except Exception as e:
            logger.exception("Error getting file info")
            raise
    
    async def delete_file(
        self,
        filename: str,
        subdirectory: str = ""
    ) -> bool:
        """
        Delete a file from storage with cache invalidation.
        
        Args:
            filename: The filename to delete
            subdirectory: Optional subdirectory within storage
            
        Returns:
            True if the file was deleted, False otherwise
        """
        try:
            file_path = self._get_file_path(filename, subdirectory)
            
            if not await aiofiles.os.path.exists(file_path):
                return False
            
            # Delete file
            await aiofiles.os.remove(file_path)
            
            # Invalidate cache
            cache_key = await self._get_cache_key("file_info", filename=filename)
            redis = await self.redis_client
            await redis.delete(cache_key)
            
            logger.info(f"File deleted: {filename}")
            return True
            
        except Exception as e:
            logger.exception("Error deleting file")
            raise
    
    def get_file_url(
        self,
        filename: str,
        subdirectory: str = ""
    ) -> str:
        """
        Get the URL for a file.
        
        Args:
            filename: The filename to get the URL for
            subdirectory: Optional subdirectory within storage
            
        Returns:
            The URL for the file
        """
        if subdirectory:
            return f"{settings.storage_url}/{subdirectory}/{filename}"
        return f"{settings.storage_url}/{filename}"
    
    async def list_files(
        self,
        subdirectory: str = "",
        pattern: str = "*",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List files in a subdirectory with optional pattern matching.
        
        Args:
            subdirectory: Optional subdirectory to list
            pattern: File pattern to match
            limit: Maximum number of files to return
            
        Returns:
            List of file information
        """
        try:
            dir_path = self._get_file_path("", subdirectory)
            
            if not await aiofiles.os.path.exists(dir_path):
                return []
            
            files = []
            count = 0
            
            async for entry in aiofiles.os.scandir(dir_path):
                if count >= limit:
                    break
                
                if entry.is_file() and Path(entry.name).match(pattern):
                    file_info = await self.get_file_info(entry.name, subdirectory)
                    if file_info:
                        files.append(file_info)
                        count += 1
            
            return files
            
        except Exception as e:
            logger.exception("Error listing files")
            raise
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files cleaned up
        """
        try:
            temp_dir = self._get_file_path("", "temp")
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            if not await aiofiles.os.path.exists(temp_dir):
                return 0
            
            async for entry in aiofiles.os.scandir(temp_dir):
                if entry.is_file():
                    stat = await aiofiles.os.stat(entry.path)
                    file_time = datetime.fromtimestamp(stat.st_ctime)
                    
                    if file_time < cutoff_time:
                        try:
                            await aiofiles.os.remove(entry.path)
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file {entry.name}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.exception("Error cleaning up temp files")
            raise
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics
        """
        try:
            stats = {
                "total_files": 0,
                "total_size": 0,
                "subdirectories": {},
                "cache_hits": 0,
                "cache_misses": 0
            }
            
            # Get cache stats
            redis = await self.redis_client
            cache_info = await redis.info("stats")
            stats["cache_hits"] = int(cache_info.get("keyspace_hits", 0))
            stats["cache_misses"] = int(cache_info.get("keyspace_misses", 0))
            
            # Get file stats
            for subdir in ["", "images", "processed"]:
                dir_path = self._get_file_path("", subdir)
                if await aiofiles.os.path.exists(dir_path):
                    dir_stats = {"files": 0, "size": 0}
                    
                    async for entry in aiofiles.os.scandir(dir_path):
                        if entry.is_file():
                            stat = await aiofiles.os.stat(entry.path)
                            dir_stats["files"] += 1
                            dir_stats["size"] += stat.st_size
                    
                    stats["subdirectories"][subdir or "root"] = dir_stats
                    stats["total_files"] += dir_stats["files"]
                    stats["total_size"] += dir_stats["size"]
            
            return stats
            
        except Exception as e:
            logger.exception("Error getting storage stats")
            raise
    
    async def close(self) -> Any:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close() 