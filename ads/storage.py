from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, BinaryIO
import os
import uuid
from datetime import datetime
import aiofiles
from fastapi import UploadFile
from onyx.utils.logger import setup_logger
from onyx.config import settings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Storage service for handling file uploads and retrievals.
"""


logger = setup_logger()

class StorageService:
    """Service for handling file storage operations."""
    
    def __init__(self) -> Any:
        """Initialize the service."""
        self.base_path = settings.STORAGE_PATH
        self.ensure_storage_path()
    
    def ensure_storage_path(self) -> Any:
        """Ensure the storage path exists."""
        os.makedirs(self.base_path, exist_ok=True)
    
    def _get_file_path(self, filename: str) -> str:
        """
        Get the full path for a file.
        
        Args:
            filename: The filename to get the path for
            
        Returns:
            The full path for the file
        """
        return os.path.join(self.base_path, filename)
    
    def _generate_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename.
        
        Args:
            original_filename: The original filename
            
        Returns:
            A unique filename
        """
        ext = os.path.splitext(original_filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}{ext}"
    
    async def save_file(
        self,
        file: BinaryIO,
        original_filename: str
    ) -> str:
        """
        Save a file to storage.
        
        Args:
            file: The file to save
            original_filename: The original filename
            
        Returns:
            The filename of the saved file
        """
        try:
            filename = self._generate_filename(original_filename)
            file_path = self._get_file_path(filename)
            
            async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(await file.read())
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            return filename
        except Exception as e:
            logger.exception("Error saving file")
            raise
    
    async async def save_upload_file(
        self,
        upload_file: UploadFile
    ) -> str:
        """
        Save an uploaded file to storage.
        
        Args:
            upload_file: The uploaded file
            
        Returns:
            The filename of the saved file
        """
        try:
            filename = self._generate_filename(upload_file.filename)
            file_path = self._get_file_path(filename)
            
            async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = await upload_file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            return filename
        except Exception as e:
            logger.exception("Error saving uploaded file")
            raise
    
    async def get_file(
        self,
        filename: str
    ) -> Optional[bytes]:
        """
        Get a file from storage.
        
        Args:
            filename: The filename to get
            
        Returns:
            The file contents if found, None otherwise
        """
        try:
            file_path = self._get_file_path(filename)
            if not os.path.exists(file_path):
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            logger.exception("Error getting file")
            raise
    
    async def delete_file(
        self,
        filename: str
    ) -> bool:
        """
        Delete a file from storage.
        
        Args:
            filename: The filename to delete
            
        Returns:
            True if the file was deleted, False otherwise
        """
        try:
            file_path = self._get_file_path(filename)
            if not os.path.exists(file_path):
                return False
            
            os.remove(file_path)
            return True
        except Exception as e:
            logger.exception("Error deleting file")
            raise
    
    def get_file_url(
        self,
        filename: str
    ) -> str:
        """
        Get the URL for a file.
        
        Args:
            filename: The filename to get the URL for
            
        Returns:
            The URL for the file
        """
        return f"{settings.STORAGE_URL}/{filename}" 