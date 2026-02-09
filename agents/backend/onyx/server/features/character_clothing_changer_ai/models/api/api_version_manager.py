"""
API Version Manager
===================

Advanced API versioning system with backward compatibility.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Version status."""
    DEPRECATED = "deprecated"
    ACTIVE = "active"
    BETA = "beta"
    STABLE = "stable"


@dataclass
class APIVersion:
    """API version."""
    version: str
    status: VersionStatus
    release_date: float
    deprecation_date: Optional[float] = None
    endpoints: List[str] = None
    breaking_changes: List[str] = None
    migration_guide: Optional[str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.breaking_changes is None:
            self.breaking_changes = []


class APIVersionManager:
    """API version manager."""
    
    def __init__(self):
        """Initialize API version manager."""
        self.versions: Dict[str, APIVersion] = {}
        self.default_version = "v1"
        self.version_handlers: Dict[str, Callable] = {}
    
    def register_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.ACTIVE,
        endpoints: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None,
        migration_guide: Optional[str] = None,
    ) -> None:
        """
        Register an API version.
        
        Args:
            version: Version string (e.g., "v1", "v2")
            status: Version status
            endpoints: List of endpoints
            breaking_changes: List of breaking changes
            migration_guide: Migration guide URL
        """
        api_version = APIVersion(
            version=version,
            status=status,
            release_date=time.time(),
            endpoints=endpoints or [],
            breaking_changes=breaking_changes or [],
            migration_guide=migration_guide,
        )
        
        self.versions[version] = api_version
        logger.info(f"Registered API version: {version} ({status.value})")
    
    def deprecate_version(
        self,
        version: str,
        deprecation_date: Optional[float] = None,
    ) -> None:
        """
        Deprecate an API version.
        
        Args:
            version: Version to deprecate
            deprecation_date: Optional deprecation date
        """
        if version not in self.versions:
            raise ValueError(f"Version not found: {version}")
        
        self.versions[version].status = VersionStatus.DEPRECATED
        self.versions[version].deprecation_date = deprecation_date or time.time()
        
        logger.warning(f"Deprecated API version: {version}")
    
    def get_version(self, version: str) -> Optional[APIVersion]:
        """
        Get version information.
        
        Args:
            version: Version string
            
        Returns:
            API version or None
        """
        return self.versions.get(version)
    
    def get_active_versions(self) -> List[APIVersion]:
        """
        Get all active versions.
        
        Returns:
            List of active versions
        """
        return [
            v for v in self.versions.values()
            if v.status in [VersionStatus.ACTIVE, VersionStatus.STABLE]
        ]
    
    def get_deprecated_versions(self) -> List[APIVersion]:
        """
        Get all deprecated versions.
        
        Returns:
            List of deprecated versions
        """
        return [
            v for v in self.versions.values()
            if v.status == VersionStatus.DEPRECATED
        ]
    
    def is_version_supported(self, version: str) -> bool:
        """
        Check if version is supported.
        
        Args:
            version: Version string
            
        Returns:
            True if supported
        """
        if version not in self.versions:
            return False
        
        api_version = self.versions[version]
        return api_version.status != VersionStatus.DEPRECATED
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get latest version.
        
        Returns:
            Latest version string
        """
        active_versions = self.get_active_versions()
        if not active_versions:
            return None
        
        # Sort by release date (newest first)
        active_versions.sort(key=lambda v: v.release_date, reverse=True)
        return active_versions[0].version
    
    def register_version_handler(
        self,
        version: str,
        handler: Callable,
    ) -> None:
        """
        Register version handler.
        
        Args:
            version: Version string
            handler: Handler function
        """
        self.version_handlers[version] = handler
        logger.debug(f"Registered version handler: {version}")
    
    def handle_request(
        self,
        version: str,
        endpoint: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Handle request for specific version.
        
        Args:
            version: Version string
            endpoint: Endpoint name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Handler result
        """
        if version not in self.version_handlers:
            raise ValueError(f"No handler for version: {version}")
        
        handler = self.version_handlers[version]
        return handler(endpoint, *args, **kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version manager statistics."""
        return {
            "total_versions": len(self.versions),
            "active_versions": len(self.get_active_versions()),
            "deprecated_versions": len(self.get_deprecated_versions()),
            "default_version": self.default_version,
            "latest_version": self.get_latest_version(),
        }

