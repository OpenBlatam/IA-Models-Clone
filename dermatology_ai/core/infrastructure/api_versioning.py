"""
API Versioning Support
Handles API versioning and backward compatibility
"""

from typing import Optional, Callable, Dict, Any
from enum import Enum
import re
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"


class APIVersionManager:
    """Manages API versions and routing"""
    
    def __init__(self, default_version: APIVersion = APIVersion.V1):
        self.default_version = default_version
        self.version_handlers: Dict[str, Dict[str, Callable]] = {}
        self.deprecated_versions: set[str] = set()
    
    def register_handler(
        self,
        endpoint: str,
        version: APIVersion,
        handler: Callable
    ):
        """
        Register a version-specific handler
        
        Args:
            endpoint: Endpoint path (e.g., "/analysis")
            version: API version
            handler: Handler function
        """
        if endpoint not in self.version_handlers:
            self.version_handlers[endpoint] = {}
        
        self.version_handlers[endpoint][version.value] = handler
        logger.debug(f"Registered handler for {endpoint} version {version.value}")
    
    def get_handler(
        self,
        endpoint: str,
        version: Optional[str] = None
    ) -> Optional[Callable]:
        """
        Get handler for endpoint and version
        
        Args:
            endpoint: Endpoint path
            version: API version (defaults to default version)
            
        Returns:
            Handler function or None
        """
        if endpoint not in self.version_handlers:
            return None
        
        version = version or self.default_version.value
        
        # Try exact version match
        if version in self.version_handlers[endpoint]:
            return self.version_handlers[endpoint][version]
        
        # Try latest
        if APIVersion.LATEST.value in self.version_handlers[endpoint]:
            return self.version_handlers[endpoint][APIVersion.LATEST.value]
        
        # Fallback to default version
        if self.default_version.value in self.version_handlers[endpoint]:
            return self.version_handlers[endpoint][self.default_version.value]
        
        return None
    
    def deprecate_version(self, version: str):
        """Mark a version as deprecated"""
        self.deprecated_versions.add(version)
        logger.info(f"Version {version} marked as deprecated")
    
    def is_deprecated(self, version: str) -> bool:
        """Check if version is deprecated"""
        return version in self.deprecated_versions


def extract_version_from_header(request) -> Optional[str]:
    """Extract API version from request header"""
    accept_header = request.headers.get("Accept", "")
    
    # Look for version in Accept header: application/vnd.api+json;version=v1
    match = re.search(r'version=([\w.]+)', accept_header)
    if match:
        return match.group(1)
    
    return None


def extract_version_from_path(path: str) -> Optional[str]:
    """Extract API version from URL path"""
    # Look for /v1/, /v2/, etc. in path
    match = re.search(r'/(v\d+)/', path)
    if match:
        return match.group(1)
    
    return None


def versioned(version: APIVersion):
    """
    Decorator to mark endpoint with API version
    
    Args:
        version: API version
    """
    def decorator(func: Callable) -> Callable:
        func._api_version = version.value
        return func
    
    return decorator


# Global version manager
_version_manager = APIVersionManager()


def get_version_manager() -> APIVersionManager:
    """Get global API version manager"""
    return _version_manager















