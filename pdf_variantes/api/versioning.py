"""
PDF Variantes API - API Versioning
API version management and compatibility
"""

from typing import List, Optional
from fastapi import Header, HTTPException
from enum import Enum


class APIVersion(str, Enum):
    """Supported API versions"""
    V1 = "v1"
    V2 = "v2"  # Future version


def get_api_version(
    api_version: Optional[str] = Header("v1", alias="API-Version")
) -> str:
    """
    Get API version from header
    
    Args:
        api_version: API version from header (default: v1)
    
    Returns:
        API version string
    
    Raises:
        HTTPException: If version is not supported
    """
    if api_version is None:
        return APIVersion.V1.value
    
    # Normalize version string
    version = api_version.lower().strip()
    if not version.startswith('v'):
        version = f"v{version}"
    
    # Check if version is supported
    try:
        APIVersion(version)
    except ValueError:
        supported = [v.value for v in APIVersion]
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported API version",
                "provided_version": api_version,
                "supported_versions": supported,
                "message": f"API version '{api_version}' is not supported. Supported versions: {', '.join(supported)}"
            }
        )
    
    return version


def check_version_compatibility(
    requested_version: str,
    required_version: str
) -> bool:
    """Check if requested version is compatible with required version"""
    if requested_version == required_version:
        return True
    
    # Version comparison logic
    # For now, only exact matches are supported
    return False


def get_version_prefix(version: str) -> str:
    """Get URL prefix for API version"""
    return f"/api/{version}"






