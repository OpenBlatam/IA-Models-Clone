"""
API Versioning for Flux2 Clothing Changer
==========================================

API version management and compatibility.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """API version status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    EXPERIMENTAL = "experimental"


@dataclass
class APIVersion:
    """API version information."""
    version: str
    status: VersionStatus
    release_date: float
    deprecation_date: Optional[float] = None
    sunset_date: Optional[float] = None
    changelog: List[str] = None
    breaking_changes: List[str] = None
    migration_guide: Optional[str] = None
    
    def __post_init__(self):
        if self.changelog is None:
            self.changelog = []
        if self.breaking_changes is None:
            self.breaking_changes = []


class APIVersioning:
    """API versioning system."""
    
    def __init__(
        self,
        default_version: str = "v1",
        enable_auto_deprecation: bool = True,
    ):
        """
        Initialize API versioning.
        
        Args:
            default_version: Default API version
            enable_auto_deprecation: Enable automatic deprecation
        """
        self.default_version = default_version
        self.enable_auto_deprecation = enable_auto_deprecation
        
        self.versions: Dict[str, APIVersion] = {}
        self.version_handlers: Dict[str, Callable] = {}
        self.compatibility_matrix: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_versions": 0,
            "active_versions": 0,
            "deprecated_versions": 0,
        }
    
    def register_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.ACTIVE,
        changelog: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None,
        migration_guide: Optional[str] = None,
    ) -> APIVersion:
        """
        Register API version.
        
        Args:
            version: Version string
            status: Version status
            changelog: Optional changelog
            breaking_changes: Optional breaking changes
            migration_guide: Optional migration guide
            
        Returns:
            Created API version
        """
        api_version = APIVersion(
            version=version,
            status=status,
            release_date=time.time(),
            changelog=changelog or [],
            breaking_changes=breaking_changes or [],
            migration_guide=migration_guide,
        )
        
        self.versions[version] = api_version
        self.stats["total_versions"] += 1
        
        if status == VersionStatus.ACTIVE:
            self.stats["active_versions"] += 1
        elif status == VersionStatus.DEPRECATED:
            self.stats["deprecated_versions"] += 1
        
        logger.info(f"Registered API version: {version} ({status.value})")
        return api_version
    
    def register_handler(
        self,
        version: str,
        handler: Callable,
    ) -> None:
        """
        Register version handler.
        
        Args:
            version: Version string
            handler: Request handler function
        """
        self.version_handlers[version] = handler
        logger.info(f"Registered handler for version: {version}")
    
    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get version information."""
        return self.versions.get(version)
    
    def is_version_supported(self, version: str) -> bool:
        """Check if version is supported."""
        if version not in self.versions:
            return False
        
        api_version = self.versions[version]
        return api_version.status in [VersionStatus.ACTIVE, VersionStatus.EXPERIMENTAL]
    
    def deprecate_version(
        self,
        version: str,
        deprecation_date: Optional[float] = None,
        sunset_date: Optional[float] = None,
    ) -> bool:
        """
        Deprecate API version.
        
        Args:
            version: Version to deprecate
            deprecation_date: Deprecation date
            sunset_date: Sunset date
            
        Returns:
            True if deprecated
        """
        if version not in self.versions:
            return False
        
        api_version = self.versions[version]
        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecation_date = deprecation_date or time.time()
        api_version.sunset_date = sunset_date
        
        self.stats["active_versions"] -= 1
        self.stats["deprecated_versions"] += 1
        
        logger.warning(f"Deprecated API version: {version}")
        return True
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest active version."""
        active_versions = [
            v for v, info in self.versions.items()
            if info.status == VersionStatus.ACTIVE
        ]
        
        if not active_versions:
            return self.default_version
        
        # Sort by release date
        active_versions.sort(
            key=lambda v: self.versions[v].release_date,
            reverse=True,
        )
        
        return active_versions[0]
    
    def get_compatible_versions(self, version: str) -> List[str]:
        """
        Get compatible versions.
        
        Args:
            version: Source version
            
        Returns:
            List of compatible versions
        """
        if version in self.compatibility_matrix:
            return self.compatibility_matrix[version]
        
        # Default: only same version
        return [version] if self.is_version_supported(version) else []
    
    def set_compatibility(
        self,
        version1: str,
        version2: str,
    ) -> None:
        """
        Set version compatibility.
        
        Args:
            version1: First version
            version2: Second version
        """
        if version1 not in self.compatibility_matrix:
            self.compatibility_matrix[version1] = []
        if version2 not in self.compatibility_matrix:
            self.compatibility_matrix[version2] = []
        
        if version2 not in self.compatibility_matrix[version1]:
            self.compatibility_matrix[version1].append(version2)
        if version1 not in self.compatibility_matrix[version2]:
            self.compatibility_matrix[version2].append(version1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        return {
            **self.stats,
            "versions": {
                v: {
                    "status": info.status.value,
                    "release_date": info.release_date,
                    "deprecation_date": info.deprecation_date,
                }
                for v, info in self.versions.items()
            },
        }


