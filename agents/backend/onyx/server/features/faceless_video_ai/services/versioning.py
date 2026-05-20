"""
Video Versioning Service
Manages versions of generated videos
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoVersion:
    """Represents a version of a video"""
    
    def __init__(
        self,
        version_id: str,
        video_id: UUID,
        version_number: int,
        video_url: str,
        config: Dict[str, Any],
        created_at: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.version_id = version_id
        self.video_id = video_id
        self.version_number = version_number
        self.video_url = video_url
        self.config = config
        self.created_at = created_at
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version_id": self.version_id,
            "video_id": str(self.video_id),
            "version_number": self.version_number,
            "video_url": self.video_url,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class VersioningService:
    """Manages video versions"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.versions: Dict[UUID, List[VideoVersion]] = {}
        self.version_counter: Dict[UUID, int] = {}
    
    def create_version(
        self,
        video_id: UUID,
        video_url: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> VideoVersion:
        """
        Create a new version of a video
        
        Args:
            video_id: Original video ID
            video_url: URL to the video file
            config: Configuration used for this version
            metadata: Additional metadata
            
        Returns:
            Created version
        """
        if video_id not in self.versions:
            self.versions[video_id] = []
            self.version_counter[video_id] = 0
        
        self.version_counter[video_id] += 1
        version_number = self.version_counter[video_id]
        
        version_id = f"{video_id}_v{version_number}"
        
        version = VideoVersion(
            version_id=version_id,
            video_id=video_id,
            version_number=version_number,
            video_url=video_url,
            config=config,
            created_at=datetime.utcnow(),
            metadata=metadata
        )
        
        self.versions[video_id].append(version)
        logger.info(f"Created version {version_number} for video {video_id}")
        
        return version
    
    def get_versions(self, video_id: UUID) -> List[VideoVersion]:
        """Get all versions of a video"""
        return self.versions.get(video_id, [])
    
    def get_version(self, video_id: UUID, version_number: int) -> Optional[VideoVersion]:
        """Get specific version"""
        versions = self.versions.get(video_id, [])
        for version in versions:
            if version.version_number == version_number:
                return version
        return None
    
    def get_latest_version(self, video_id: UUID) -> Optional[VideoVersion]:
        """Get latest version of a video"""
        versions = self.versions.get(video_id, [])
        if not versions:
            return None
        return max(versions, key=lambda v: v.version_number)
    
    def compare_versions(
        self,
        video_id: UUID,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(video_id, version1)
        v2 = self.get_version(video_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        differences = []
        
        # Compare configs
        for key in set(v1.config.keys()) | set(v2.config.keys()):
            val1 = v1.config.get(key)
            val2 = v2.config.get(key)
            if val1 != val2:
                differences.append({
                    "field": key,
                    "version1": val1,
                    "version2": val2,
                })
        
        return {
            "version1": v1.to_dict(),
            "version2": v2.to_dict(),
            "differences": differences,
        }


_versioning_service: Optional[VersioningService] = None


def get_versioning_service() -> VersioningService:
    """Get versioning service instance (singleton)"""
    global _versioning_service
    if _versioning_service is None:
        _versioning_service = VersioningService()
    return _versioning_service

