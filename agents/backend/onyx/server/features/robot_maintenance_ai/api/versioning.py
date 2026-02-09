"""
API versioning utilities.
"""

from fastapi import APIRouter, Request
from typing import Dict, Any
import re

API_VERSION_PATTERN = re.compile(r'^v(\d+)\.(\d+)$')


class APIVersion:
    """Represents an API version."""
    
    def __init__(self, major: int, minor: int):
        self.major = major
        self.minor = minor
    
    def __str__(self):
        return f"v{self.major}.{self.minor}"
    
    def __eq__(self, other):
        return self.major == other.major and self.minor == other.minor
    
    def __lt__(self, other):
        if self.major < other.major:
            return True
        if self.major == other.major:
            return self.minor < other.minor
        return False
    
    @classmethod
    def from_string(cls, version_str: str) -> 'APIVersion':
        """
        Parse version string.
        
        Args:
            version_str: Version string (e.g., "v1.0")
        
        Returns:
            APIVersion instance
        """
        match = API_VERSION_PATTERN.match(version_str)
        if match:
            return cls(int(match.group(1)), int(match.group(2)))
        raise ValueError(f"Invalid version format: {version_str}")


def get_api_version(request: Request) -> APIVersion:
    """
    Extract API version from request.
    
    Args:
        request: FastAPI request
    
    Returns:
        API version
    """
    version_header = request.headers.get("API-Version", "v1.0")
    try:
        return APIVersion.from_string(version_header)
    except ValueError:
        return APIVersion(1, 0)


CURRENT_API_VERSION = APIVersion(1, 0)


def create_versioned_router(version: APIVersion) -> APIRouter:
    """
    Create a versioned router.
    
    Args:
        version: API version
    
    Returns:
        Versioned router
    """
    return APIRouter(
        prefix=f"/api/v{version.major}.{version.minor}/robot-maintenance",
        tags=[f"Robot Maintenance AI v{version}"]
    )






