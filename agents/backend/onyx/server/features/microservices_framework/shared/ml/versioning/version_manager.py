"""
Version Manager
Version management for models and services.
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Version type."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRE_RELEASE = "pre-release"
    BUILD = "build"


@dataclass
class Version:
    """Semantic version."""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: "Version") -> bool:
        """Compare versions."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        return False
    
    def __eq__(self, other: "Version") -> bool:
        """Check equality."""
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.pre_release == other.pre_release
            and self.build == other.build
        )


class VersionManager:
    """Manage versions."""
    
    @staticmethod
    def parse(version_str: str) -> Version:
        """Parse version string."""
        # Semantic versioning: MAJOR.MINOR.PATCH[-PRE_RELEASE][+BUILD]
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([\w\.-]+))?(?:\+([\w\.-]+))?$"
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor, patch, pre_release, build = match.groups()
        
        return Version(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            pre_release=pre_release,
            build=build,
        )
    
    @staticmethod
    def increment(
        version: Version,
        version_type: VersionType,
        pre_release: Optional[str] = None,
        build: Optional[str] = None,
    ) -> Version:
        """Increment version."""
        if version_type == VersionType.MAJOR:
            return Version(
                major=version.major + 1,
                minor=0,
                patch=0,
                pre_release=pre_release,
                build=build,
            )
        elif version_type == VersionType.MINOR:
            return Version(
                major=version.major,
                minor=version.minor + 1,
                patch=0,
                pre_release=pre_release,
                build=build,
            )
        elif version_type == VersionType.PATCH:
            return Version(
                major=version.major,
                minor=version.minor,
                patch=version.patch + 1,
                pre_release=pre_release,
                build=build,
            )
        elif version_type == VersionType.PRE_RELEASE:
            return Version(
                major=version.major,
                minor=version.minor,
                patch=version.patch,
                pre_release=pre_release or "alpha.1",
                build=build,
            )
        elif version_type == VersionType.BUILD:
            return Version(
                major=version.major,
                minor=version.minor,
                patch=version.patch,
                pre_release=version.pre_release,
                build=build or "1",
            )
        else:
            raise ValueError(f"Unknown version type: {version_type}")
    
    @staticmethod
    def compare(version1: str, version2: str) -> int:
        """Compare two versions. Returns -1, 0, or 1."""
        v1 = VersionManager.parse(version1)
        v2 = VersionManager.parse(version2)
        
        if v1 < v2:
            return -1
        elif v1 == v2:
            return 0
        else:
            return 1
    
    @staticmethod
    def get_latest(versions: List[str]) -> str:
        """Get latest version from list."""
        if not versions:
            raise ValueError("Empty version list")
        
        sorted_versions = sorted(versions, key=lambda v: VersionManager.parse(v), reverse=True)
        return sorted_versions[0]
    
    @staticmethod
    def is_compatible(version1: str, version2: str, level: str = "patch") -> bool:
        """Check if versions are compatible."""
        v1 = VersionManager.parse(version1)
        v2 = VersionManager.parse(version2)
        
        if level == "major":
            return v1.major == v2.major
        elif level == "minor":
            return v1.major == v2.major and v1.minor == v2.minor
        elif level == "patch":
            return v1.major == v2.major and v1.minor == v2.minor and v1.patch == v2.patch
        else:
            raise ValueError(f"Unknown compatibility level: {level}")



