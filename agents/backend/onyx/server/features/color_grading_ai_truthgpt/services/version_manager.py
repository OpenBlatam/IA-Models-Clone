"""
Version Manager for Color Grading AI
=====================================

Manages versioning of color grading parameters and results.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """Version data structure."""
    version_id: str
    parent_id: Optional[str]  # Parent version ID
    color_params: Dict[str, Any]
    created_at: datetime
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_current: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class VersionManager:
    """
    Manages versioning of color grading parameters.
    
    Features:
    - Create versions
    - Branch and merge
    - Compare versions
    - Rollback to previous version
    """
    
    def __init__(self, versions_dir: str = "versions"):
        """
        Initialize version manager.
        
        Args:
            versions_dir: Directory for versions storage
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, List[Version]] = {}  # media_id -> versions
    
    def create_version(
        self,
        media_id: str,
        color_params: Dict[str, Any],
        parent_id: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new version.
        
        Args:
            media_id: Media identifier
            color_params: Color parameters
            parent_id: Parent version ID
            description: Version description
            created_by: Creator identifier
            tags: Optional tags
            
        Returns:
            Version ID
        """
        import uuid
        version_id = str(uuid.uuid4())
        
        # Mark all previous versions as not current
        if media_id in self._versions:
            for version in self._versions[media_id]:
                version.is_current = False
        
        version = Version(
            version_id=version_id,
            parent_id=parent_id,
            color_params=color_params,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            is_current=True
        )
        
        if media_id not in self._versions:
            self._versions[media_id] = []
        
        self._versions[media_id].append(version)
        self._save_versions(media_id)
        
        logger.info(f"Created version {version_id} for media {media_id}")
        return version_id
    
    def get_versions(self, media_id: str) -> List[Version]:
        """
        Get all versions for media.
        
        Args:
            media_id: Media identifier
            
        Returns:
            List of versions
        """
        return self._versions.get(media_id, [])
    
    def get_current_version(self, media_id: str) -> Optional[Version]:
        """
        Get current version for media.
        
        Args:
            media_id: Media identifier
            
        Returns:
            Current version or None
        """
        versions = self.get_versions(media_id)
        for version in versions:
            if version.is_current:
                return version
        return None
    
    def get_version(self, media_id: str, version_id: str) -> Optional[Version]:
        """
        Get specific version.
        
        Args:
            media_id: Media identifier
            version_id: Version ID
            
        Returns:
            Version or None
        """
        versions = self.get_versions(media_id)
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    def rollback_to_version(self, media_id: str, version_id: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            media_id: Media identifier
            version_id: Version ID to rollback to
            
        Returns:
            True if successful
        """
        version = self.get_version(media_id, version_id)
        if not version:
            return False
        
        # Mark all as not current
        for v in self._versions[media_id]:
            v.is_current = False
        
        # Mark target as current
        version.is_current = True
        self._save_versions(media_id)
        
        logger.info(f"Rolled back to version {version_id} for media {media_id}")
        return True
    
    def compare_versions(
        self,
        media_id: str,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            media_id: Media identifier
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Comparison result
        """
        version1 = self.get_version(media_id, version_id1)
        version2 = self.get_version(media_id, version_id2)
        
        if not version1 or not version2:
            return {"error": "Version not found"}
        
        # Calculate differences
        params1 = version1.color_params
        params2 = version2.color_params
        
        differences = {}
        for key in set(list(params1.keys()) + list(params2.keys())):
            val1 = params1.get(key)
            val2 = params2.get(key)
            if val1 != val2:
                differences[key] = {
                    "version1": val1,
                    "version2": val2,
                    "diff": self._calculate_diff(val1, val2)
                }
        
        return {
            "version1": version1.to_dict(),
            "version2": version2.to_dict(),
            "differences": differences,
        }
    
    def _calculate_diff(self, val1: Any, val2: Any) -> Any:
        """Calculate difference between two values."""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return val2 - val1
        elif isinstance(val1, dict) and isinstance(val2, dict):
            return {k: v2 - v1 for k, v1, v2 in zip(val1.keys(), val1.values(), val2.values()) if isinstance(v1, (int, float))}
        return None
    
    def _save_versions(self, media_id: str):
        """Save versions to disk."""
        versions_file = self.versions_dir / f"{media_id}.json"
        data = {
            "media_id": media_id,
            "versions": [v.to_dict() for v in self._versions[media_id]]
        }
        with open(versions_file, "w") as f:
            json.dump(data, f, indent=2, default=str)




