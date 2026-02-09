"""
Version Manager
===============

Advanced version management for APIs and data structures.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import semver

logger = logging.getLogger(__name__)

class VersionStrategy(str, Enum):
    """Versioning strategies."""
    SEMANTIC = "semantic"
    DATE = "date"
    INCREMENTAL = "incremental"

@dataclass
class APIVersion:
    """API version definition."""
    version: str
    strategy: VersionStrategy
    released_at: datetime
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None
    end_of_life: Optional[datetime] = None
    changelog: List[str] = None
    migration_guide: Optional[str] = None

class VersionManager:
    """Advanced version manager."""
    
    def __init__(self):
        self.versions: Dict[str, APIVersion] = {}
        self.current_version: Optional[str] = None
        self.default_version: Optional[str] = None
    
    def register_version(
        self,
        version: str,
        strategy: VersionStrategy = VersionStrategy.SEMANTIC,
        released_at: Optional[datetime] = None,
        changelog: Optional[List[str]] = None,
        migration_guide: Optional[str] = None
    ) -> APIVersion:
        """Register a new version."""
        api_version = APIVersion(
            version=version,
            strategy=strategy,
            released_at=released_at or datetime.now(),
            changelog=changelog or [],
            migration_guide=migration_guide
        )
        
        self.versions[version] = api_version
        
        if not self.current_version:
            self.current_version = version
            self.default_version = version
        
        logger.info(f"Version registered: {version}")
        
        return api_version
    
    def deprecate_version(
        self,
        version: str,
        deprecated_at: Optional[datetime] = None,
        end_of_life: Optional[datetime] = None
    ):
        """Deprecate a version."""
        if version in self.versions:
            self.versions[version].deprecated = True
            self.versions[version].deprecated_at = deprecated_at or datetime.now()
            self.versions[version].end_of_life = end_of_life
            logger.info(f"Version deprecated: {version}")
    
    def set_current_version(self, version: str):
        """Set current version."""
        if version in self.versions:
            self.current_version = version
            logger.info(f"Current version set to: {version}")
    
    def set_default_version(self, version: str):
        """Set default version."""
        if version in self.versions:
            self.default_version = version
            logger.info(f"Default version set to: {version}")
    
    def get_version(self, version: Optional[str] = None) -> Optional[APIVersion]:
        """Get version information."""
        if version is None:
            version = self.default_version
        
        return self.versions.get(version)
    
    def is_deprecated(self, version: str) -> bool:
        """Check if version is deprecated."""
        api_version = self.versions.get(version)
        return api_version.deprecated if api_version else False
    
    def is_supported(self, version: str) -> bool:
        """Check if version is supported."""
        api_version = self.versions.get(version)
        if not api_version:
            return False
        
        if api_version.deprecated:
            if api_version.end_of_life:
                return datetime.now() < api_version.end_of_life
            return True
        
        return True
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """Compare two versions."""
        try:
            v1 = semver.VersionInfo.parse(version1)
            v2 = semver.VersionInfo.parse(version2)
            
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
            else:
                return 0
        except:
            # Fallback to string comparison
            if version1 > version2:
                return 1
            elif version1 < version2:
                return -1
            else:
                return 0
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version."""
        if not self.versions:
            return None
        
        versions = list(self.versions.keys())
        
        # Try semantic versioning
        try:
            sorted_versions = sorted(
                versions,
                key=lambda v: semver.VersionInfo.parse(v),
                reverse=True
            )
            return sorted_versions[0]
        except:
            # Fallback to string sort
            return sorted(versions, reverse=True)[0]
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """Get all versions."""
        return [
            {
                "version": v.version,
                "strategy": v.strategy.value,
                "released_at": v.released_at.isoformat(),
                "deprecated": v.deprecated,
                "deprecated_at": v.deprecated_at.isoformat() if v.deprecated_at else None,
                "end_of_life": v.end_of_life.isoformat() if v.end_of_life else None,
                "changelog": v.changelog,
                "supported": self.is_supported(v.version)
            }
            for v in self.versions.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get version statistics."""
        total = len(self.versions)
        deprecated = sum(1 for v in self.versions.values() if v.deprecated)
        supported = sum(1 for v in self.versions.values() if self.is_supported(v.version))
        
        return {
            "total_versions": total,
            "deprecated_versions": deprecated,
            "supported_versions": supported,
            "current_version": self.current_version,
            "default_version": self.default_version,
            "latest_version": self.get_latest_version()
        }

# Global instance
version_manager = VersionManager()

















