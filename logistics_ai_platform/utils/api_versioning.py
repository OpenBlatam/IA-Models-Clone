"""
API Versioning utilities

This module provides utilities for API versioning support.
"""

from typing import Optional
from fastapi import Header, HTTPException, status
from enum import Enum


class APIVersion(str, Enum):
    """Supported API versions"""
    V1 = "v1"
    V2 = "v2"  # Future version


CURRENT_API_VERSION = APIVersion.V1
DEFAULT_API_VERSION = APIVersion.V1


def get_api_version(
    accept_version: Optional[str] = Header(None, alias="Accept-Version"),
    api_version: Optional[str] = Header(None, alias="X-API-Version")
) -> APIVersion:
    """
    Extract API version from request headers
    
    Args:
        accept_version: Version from Accept-Version header
        api_version: Version from X-API-Version header
        
    Returns:
        APIVersion enum value
        
    Raises:
        HTTPException: If version is invalid or unsupported
    """
    version_str = api_version or accept_version
    
    if version_str is None:
        return DEFAULT_API_VERSION
    
    # Normalize version string (remove 'v' prefix if present, lowercase)
    version_str = version_str.lower().lstrip('v')
    
    # Map to enum
    version_map = {
        "1": APIVersion.V1,
        "v1": APIVersion.V1,
        "2": APIVersion.V2,
        "v2": APIVersion.V2,
    }
    
    if version_str not in version_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported API version: {version_str}. Supported versions: v1"
        )
    
    requested_version = version_map[version_str]
    
    # Check if version is available
    if requested_version == APIVersion.V2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API version v2 is not yet available. Please use v1."
        )
    
    return requested_version


def validate_api_version(version: APIVersion) -> bool:
    """
    Validate if API version is supported
    
    Args:
        version: API version to validate
        
    Returns:
        True if version is supported
    """
    return version in APIVersion


def get_versioned_path(base_path: str, version: Optional[APIVersion] = None) -> str:
    """
    Get versioned API path
    
    Args:
        base_path: Base API path
        version: API version (defaults to current version)
        
    Returns:
        Versioned API path
    """
    if version is None:
        version = CURRENT_API_VERSION
    
    return f"/api/{version.value}{base_path}"

