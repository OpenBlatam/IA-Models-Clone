"""
Version Manager
==============

API version management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Version status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


@dataclass
class APIVersion:
    """API version definition."""
    version: str
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changelog: List[str] = None
    
    def __post_init__(self):
        if self.changelog is None:
            self.changelog = []


class VersionManager:
    """API version manager."""
    
    def __init__(self):
        self._versions: Dict[str, APIVersion] = {}
        self._default_version: Optional[str] = None
    
    def register_version(
        self,
        version: str,
        release_date: datetime,
        changelog: Optional[List[str]] = None,
        is_default: bool = False
    ) -> APIVersion:
        """Register API version."""
        api_version = APIVersion(
            version=version,
            status=VersionStatus.ACTIVE,
            release_date=release_date,
            changelog=changelog or []
        )
        
        self._versions[version] = api_version
        
        if is_default or not self._default_version:
            self._default_version = version
        
        logger.info(f"Registered API version: {version}")
        return api_version
    
    def deprecate_version(self, version: str, deprecation_date: datetime):
        """Deprecate version."""
        if version not in self._versions:
            return False
        
        api_version = self._versions[version]
        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecation_date = deprecation_date
        
        logger.warning(f"Deprecated API version: {version}")
        return True
    
    def sunset_version(self, version: str, sunset_date: datetime):
        """Sunset version."""
        if version not in self._versions:
            return False
        
        api_version = self._versions[version]
        api_version.status = VersionStatus.SUNSET
        api_version.sunset_date = sunset_date
        
        logger.warning(f"Sunset API version: {version}")
        return True
    
    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get version info."""
        return self._versions.get(version)
    
    def get_active_versions(self) -> List[APIVersion]:
        """Get active versions."""
        return [
            v for v in self._versions.values()
            if v.status == VersionStatus.ACTIVE
        ]
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version."""
        active = self.get_active_versions()
        if not active:
            return self._default_version
        
        # Sort by release date
        active.sort(key=lambda x: x.release_date, reverse=True)
        return active[0].version
    
    def set_default_version(self, version: str):
        """Set default version."""
        if version in self._versions:
            self._default_version = version
            logger.info(f"Set default version: {version}")
    
    def get_default_version(self) -> Optional[str]:
        """Get default version."""
        return self._default_version
    
    def get_version_stats(self) -> Dict[str, Any]:
        """Get version statistics."""
        return {
            "total_versions": len(self._versions),
            "active_versions": len(self.get_active_versions()),
            "deprecated_versions": sum(
                1 for v in self._versions.values()
                if v.status == VersionStatus.DEPRECATED
            ),
            "sunset_versions": sum(
                1 for v in self._versions.values()
                if v.status == VersionStatus.SUNSET
            ),
            "default_version": self._default_version,
            "latest_version": self.get_latest_version()
        }















