"""
Storage infrastructure for the ads feature.

This module consolidates storage functionality from:
- storage.py (basic file storage)
- optimized_storage.py (production storage with caching)

Provides unified storage management with strategy pattern for different backends.
"""

import os
import uuid
import hashlib
import aiofiles
import aiofiles.os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union
from enum import Enum

try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from fastapi import UploadFile, HTTPException

try:
    from onyx.utils.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback minimal logger for tests
    import logging as _logging

    def setup_logger(name: str | None = None):  # type: ignore[override]
        logger = _logging.getLogger(name or __name__)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger
from ..config import get_optimized_settings

logger = setup_logger()

class StorageType(Enum):
    """Storage backend types."""
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

@dataclass
class StorageConfig:
    """Storage configuration settings."""
    base_path: str = "storage"
    cache_ttl: int = 3600
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "pdf", "doc", "docx"
    ])
    storage_type: StorageType = StorageType.LOCAL
    cloud_provider: Optional[str] = None
    cloud_credentials: Optional[Dict[str, str]] = None
    compression_enabled: bool = True
    encryption_enabled: bool = False

class StorageStrategy(ABC):
    """Abstract storage strategy interface."""
    
    @abstractmethod
    async def save_file(self, file: BinaryIO, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a file using the strategy."""
        pass
    
    @abstractmethod
    async def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve a file using the strategy."""
        pass
    
    @abstractmethod
    async def delete_file(self, filename: str) -> bool:
        """Delete a file using the strategy."""
        pass
    
    @abstractmethod
    async def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get file information using the strategy."""
        pass
    
    @abstractmethod
    async def list_files(self, pattern: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
        """List files using the strategy."""
        pass

class LocalStorageStrategy(StorageStrategy):
    """Local file system storage strategy."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Ensure storage directories exist."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for better organization
            (self.base_path / "images").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            (self.base_path / "processed").mkdir(exist_ok=True)
            (self.base_path / "documents").mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            raise
    
    def _get_file_path(self, filename: str, subdirectory: str = "") -> Path:
        """Get the full path for a file."""
        if subdirectory:
            return self.base_path / subdirectory / filename
        return self.base_path / filename
    
    def _generate_filename(self, original_filename: str, prefix: str = "") -> str:
        """Generate a unique filename."""
        ext = Path(original_filename).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        prefix_part = f"{prefix}_" if prefix else ""
        return f"{prefix_part}{timestamp}_{unique_id}{ext}"
    
    def _validate_file_type(self, filename: str) -> bool:
        """Validate if file type is allowed."""
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in self.config.allowed_extensions
    
    def _validate_file_size(self, file_size: int) -> bool:
        """Validate if file size is within limits."""
        return file_size <= self.config.max_file_size
    
    async def save_file(self, file: BinaryIO, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a file to local storage."""
        try:
            # Validate file
            if not self._validate_file_type(filename):
                raise HTTPException(status_code=400, detail="File type not allowed")
            
            # Generate unique filename
            unique_filename = self._generate_filename(filename)
            file_path = self._get_file_path(unique_filename)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                if not self._validate_file_size(len(content)):
                    raise HTTPException(status_code=400, detail="File size exceeds limit")
                await f.write(content)
            
            # Save metadata if provided
            if metadata:
                metadata_path = file_path.with_suffix('.json')
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(str(metadata))
            
            logger.info(f"File saved successfully: {unique_filename}")
            return unique_filename
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    async def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve a file from local storage."""
        try:
            file_path = self._get_file_path(filename)
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return None
    
    async def delete_file(self, filename: str) -> bool:
        """Delete a file from local storage."""
        try:
            file_path = self._get_file_path(filename)
            if file_path.exists():
                file_path.unlink()
                
                # Delete metadata if exists
                metadata_path = file_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"File deleted successfully: {filename}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False
    
    async def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get file information from local storage."""
        try:
            file_path = self._get_file_path(filename)
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            file_info = {
                "filename": filename,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "path": str(file_path)
            }
            
            # Try to load metadata
            metadata_path = file_path.with_suffix('.json')
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_content = await f.read()
                    try:
                        import json
                        file_info["metadata"] = json.loads(metadata_content)
                    except:
                        file_info["metadata"] = metadata_content
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting file info for {filename}: {e}")
            return None
    
    async def list_files(self, pattern: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
        """List files in local storage."""
        try:
            files = []
            for file_path in self.base_path.rglob(pattern):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    file_info = await self.get_file_info(file_path.name)
                    if file_info:
                        files.append(file_info)
                        if len(files) >= limit:
                            break
            
            return sorted(files, key=lambda x: x["modified_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

class CloudStorageStrategy(StorageStrategy):
    """Cloud storage strategy (placeholder for future implementation)."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.provider = config.cloud_provider
        self.credentials = config.cloud_credentials or {}
    
    async def save_file(self, file: BinaryIO, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a file to cloud storage."""
        # TODO: Implement cloud storage integration
        logger.warning("Cloud storage not yet implemented")
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve a file from cloud storage."""
        # TODO: Implement cloud storage integration
        logger.warning("Cloud storage not yet implemented")
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def delete_file(self, filename: str) -> bool:
        """Delete a file from cloud storage."""
        # TODO: Implement cloud storage integration
        logger.warning("Cloud storage not yet implemented")
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get file information from cloud storage."""
        # TODO: Implement cloud storage integration
        logger.warning("Cloud storage not yet implemented")
        raise NotImplementedError("Cloud storage not yet implemented")
    
    async def list_files(self, pattern: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
        """List files in cloud storage."""
        # TODO: Implement cloud storage integration
        logger.warning("Cloud storage not yet implemented")
        raise NotImplementedError("Cloud storage not yet implemented")

class FileStorageManager:
    """Manages file storage operations with strategy pattern."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        if config is None:
            settings = get_optimized_settings()
            config = StorageConfig(
                base_path=settings.storage_path,
                cache_ttl=settings.image_cache_ttl,
                max_file_size=settings.max_file_size,
                allowed_extensions=settings.allowed_file_types,
                storage_type=StorageType.LOCAL
            )
        
        self.config = config
        self.strategy = self._create_strategy()
        self._redis_client = None
    
    def _create_strategy(self) -> StorageStrategy:
        """Create storage strategy based on configuration."""
        if self.config.storage_type == StorageType.LOCAL:
            return LocalStorageStrategy(self.config)
        elif self.config.storage_type == StorageType.CLOUD:
            return CloudStorageStrategy(self.config)
        else:
            # Default to local storage
            return LocalStorageStrategy(self.config)
    
    @property
    async def redis_client(self):
        """Get Redis client for caching."""
        if self._redis_client is None:
            settings = get_optimized_settings()
            if aioredis is None or not getattr(settings, "redis_url", None):
                self._redis_client = None
            else:
                self._redis_client = await aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
        return self._redis_client
    
    async def save_file(self, file: BinaryIO, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a file using the configured strategy."""
        return await self.strategy.save_file(file, filename, metadata)
    
    async def save_upload_file(self, upload_file: UploadFile, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save an uploaded file."""
        try:
            # Validate file
            if not self._validate_upload_file(upload_file):
                raise HTTPException(status_code=400, detail="Invalid upload file")
            
            # Save file
            filename = await self.strategy.save_file(upload_file.file, upload_file.filename, metadata)
            
            # Cache file info
            await self._cache_file_info(filename, {
                "original_filename": upload_file.filename,
                "content_type": upload_file.content_type,
                "size": upload_file.size,
                "metadata": metadata,
                "uploaded_at": datetime.now().isoformat()
            })
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise
    
    def _validate_upload_file(self, upload_file: UploadFile) -> bool:
        """Validate uploaded file."""
        if not upload_file.filename:
            return False
        
        if not self._validate_file_type(upload_file.filename):
            return False
        
        if upload_file.size and upload_file.size > self.config.max_file_size:
            return False
        
        return True
    
    def _validate_file_type(self, filename: str) -> bool:
        """Validate file type."""
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in self.config.allowed_extensions
    
    async def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve a file using the configured strategy."""
        return await self.strategy.get_file(filename)
    
    async def delete_file(self, filename: str) -> bool:
        """Delete a file using the configured strategy."""
        # Invalidate cache
        await self._invalidate_file_cache(filename)
        
        # Delete file
        return await self.strategy.delete_file(filename)
    
    async def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get file information using the configured strategy."""
        # Try cache first
        cached_info = await self._get_cached_file_info(filename)
        if cached_info:
            return cached_info
        
        # Get from storage
        file_info = await self.strategy.get_file_info(filename)
        if file_info:
            # Cache the result
            await self._cache_file_info(filename, file_info)
        
        return file_info
    
    async def list_files(self, pattern: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
        """List files using the configured strategy."""
        return await self.strategy.list_files(pattern, limit)
    
    async def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for file operations."""
        data = str(operation) + str(sorted(kwargs.items()))
        return f"storage:{hashlib.md5(data.encode()).hexdigest()}"
    
    async def _get_cached_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get cached file information."""
        try:
            cache_key = await self._get_cache_key("file_info", filename=filename)
            redis_client = await self.redis_client
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                import json
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_file_info(self, filename: str, file_info: Dict[str, Any], ttl: int = None):
        """Cache file information."""
        try:
            cache_key = await self._get_cache_key("file_info", filename=filename)
            redis_client = await self.redis_client
            ttl = ttl or self.config.cache_ttl
            import json
            await redis_client.setex(cache_key, ttl, json.dumps(file_info))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _invalidate_file_cache(self, filename: str):
        """Invalidate file cache."""
        try:
            cache_key = await self._get_cache_key("file_info", filename=filename)
            redis_client = await self.redis_client
            await redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified age."""
        try:
            temp_dir = self.base_path / "temp"
            if not temp_dir.exists():
                return 0
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                "total_files": 0,
                "total_size": 0,
                "file_types": {},
                "storage_path": str(self.base_path),
                "strategy": self.config.storage_type.value
            }
            
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    stats["total_files"] += 1
                    stats["total_size"] += file_path.stat().st_size
                    
                    ext = file_path.suffix.lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    async def close(self):
        """Close storage manager."""
        if self._redis_client:
            await self._redis_client.close()

# Alias for backward compatibility
StorageService = FileStorageManager
