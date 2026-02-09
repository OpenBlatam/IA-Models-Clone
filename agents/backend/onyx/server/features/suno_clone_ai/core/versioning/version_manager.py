"""
Version Management

Utilities for managing versions.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class VersionManager:
    """Manage versions and version strings."""
    
    @staticmethod
    def create_version(
        major: int = 1,
        minor: int = 0,
        patch: int = 0,
        suffix: Optional[str] = None
    ) -> str:
        """
        Create version string.
        
        Args:
            major: Major version
            minor: Minor version
            patch: Patch version
            suffix: Optional suffix (e.g., 'dev', 'alpha', 'beta')
            
        Returns:
            Version string
        """
        version = f"{major}.{minor}.{patch}"
        if suffix:
            version = f"{version}-{suffix}"
        return version
    
    @staticmethod
    def parse_version(version: str) -> Dict[str, Any]:
        """
        Parse version string.
        
        Args:
            version: Version string
            
        Returns:
            Parsed version components
        """
        # Match semantic versioning pattern
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([\w]+))?$'
        match = re.match(pattern, version)
        
        if match:
            return {
                'major': int(match.group(1)),
                'minor': int(match.group(2)),
                'patch': int(match.group(3)),
                'suffix': match.group(4)
            }
        else:
            raise ValueError(f"Invalid version format: {version}")
    
    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        v1 = VersionManager.parse_version(version1)
        v2 = VersionManager.parse_version(version2)
        
        if v1['major'] != v2['major']:
            return -1 if v1['major'] < v2['major'] else 1
        
        if v1['minor'] != v2['minor']:
            return -1 if v1['minor'] < v2['minor'] else 1
        
        if v1['patch'] != v2['patch']:
            return -1 if v1['patch'] < v2['patch'] else 1
        
        return 0


def create_version(**kwargs) -> str:
    """Create version string."""
    return VersionManager.create_version(**kwargs)


def get_version(version: str) -> Dict[str, Any]:
    """Parse version string."""
    return VersionManager.parse_version(version)


def compare_versions(version1: str, version2: str) -> int:
    """Compare versions."""
    return VersionManager.compare_versions(version1, version2)



