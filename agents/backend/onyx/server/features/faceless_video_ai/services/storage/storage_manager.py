"""
Storage Manager
Manages file storage (local and cloud)
"""

from typing import Optional
from pathlib import Path
import logging

from .s3_storage import get_s3_storage

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage with cloud backup"""
    
    def __init__(self, use_cloud: bool = True):
        self.use_cloud = use_cloud
        self.s3_storage = get_s3_storage() if use_cloud else None
    
    def upload_video(
        self,
        video_path: Path,
        video_id: str,
        upload_to_cloud: bool = True
    ) -> dict:
        """
        Upload video to storage
        
        Args:
            video_path: Local video file path
            video_id: Video ID
            upload_to_cloud: Whether to upload to cloud
            
        Returns:
            Dictionary with local and cloud URLs
        """
        result = {
            "local_path": str(video_path),
            "cloud_url": None,
        }
        
        if upload_to_cloud and self.s3_storage and self.s3_storage.is_available():
            s3_key = f"videos/{video_id}.mp4"
            cloud_url = self.s3_storage.upload_file(
                video_path,
                s3_key,
                content_type="video/mp4"
            )
            result["cloud_url"] = cloud_url
        
        return result
    
    def upload_thumbnail(
        self,
        thumbnail_path: Path,
        video_id: str,
        upload_to_cloud: bool = True
    ) -> dict:
        """
        Upload thumbnail to storage
        
        Args:
            thumbnail_path: Local thumbnail file path
            video_id: Video ID
            upload_to_cloud: Whether to upload to cloud
            
        Returns:
            Dictionary with local and cloud URLs
        """
        result = {
            "local_path": str(thumbnail_path),
            "cloud_url": None,
        }
        
        if upload_to_cloud and self.s3_storage and self.s3_storage.is_available():
            s3_key = f"thumbnails/{video_id}.jpg"
            cloud_url = self.s3_storage.upload_file(
                thumbnail_path,
                s3_key,
                content_type="image/jpeg"
            )
            result["cloud_url"] = cloud_url
        
        return result
    
    def delete_video(self, video_id: str, delete_from_cloud: bool = True) -> bool:
        """Delete video from storage"""
        if delete_from_cloud and self.s3_storage and self.s3_storage.is_available():
            s3_key = f"videos/{video_id}.mp4"
            return self.s3_storage.delete_file(s3_key)
        return True


_storage_manager: Optional[StorageManager] = None


def get_storage_manager(use_cloud: bool = True) -> StorageManager:
    """Get storage manager instance (singleton)"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager(use_cloud=use_cloud)
    return _storage_manager

