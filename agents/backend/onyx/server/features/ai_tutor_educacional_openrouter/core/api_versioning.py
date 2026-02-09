"""
API versioning system for backward compatibility.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


class APIVersionManager:
    """
    Manages API versioning and backward compatibility.
    """
    
    def __init__(self):
        self.current_version = APIVersion.LATEST
        self.supported_versions = [APIVersion.V1, APIVersion.V2]
        self.deprecated_versions = []
    
    def get_version_from_header(self, headers: Dict[str, str]) -> APIVersion:
        """Extract API version from request headers."""
        version_header = headers.get("X-API-Version", APIVersion.LATEST.value)
        
        try:
            return APIVersion(version_header.lower())
        except ValueError:
            logger.warning(f"Unknown API version: {version_header}, using {APIVersion.LATEST.value}")
            return APIVersion.LATEST
    
    def get_version_from_path(self, path: str) -> Optional[APIVersion]:
        """Extract API version from URL path."""
        if "/v1/" in path:
            return APIVersion.V1
        elif "/v2/" in path:
            return APIVersion.V2
        return None
    
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if version is supported."""
        return version in self.supported_versions
    
    def is_version_deprecated(self, version: APIVersion) -> bool:
        """Check if version is deprecated."""
        return version in self.deprecated_versions
    
    def transform_response(
        self,
        data: Dict[str, Any],
        requested_version: APIVersion,
        current_version: APIVersion = None
    ) -> Dict[str, Any]:
        """
        Transform response to match requested version.
        
        Args:
            data: Response data
            requested_version: Version requested by client
            current_version: Current API version (defaults to latest)
        
        Returns:
            Transformed response data
        """
        if current_version is None:
            current_version = self.current_version
        
        if requested_version == current_version:
            return data
        
        # Version transformation logic
        if requested_version == APIVersion.V1:
            return self._transform_to_v1(data)
        elif requested_version == APIVersion.V2:
            return self._transform_to_v2(data)
        
        return data
    
    def _transform_to_v1(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data to v1 format."""
        # Example transformation
        if "data" in data:
            return {
                "result": data["data"],
                "status": "success" if data.get("success") else "error"
            }
        return data
    
    def _transform_to_v2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data to v2 format."""
        # v2 is current, no transformation needed
        return data
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get information about API versions."""
        return {
            "current_version": self.current_version.value,
            "supported_versions": [v.value for v in self.supported_versions],
            "deprecated_versions": [v.value for v in self.deprecated_versions],
            "versioning_strategy": "header_and_path"
        }






