"""
API versioning utilities
"""

from typing import Optional
from fastapi import APIRouter, Header
from enum import Enum


class APIVersion(str, Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


def get_api_version(version_header: Optional[str] = Header(None, alias="X-API-Version")) -> APIVersion:
    """
    Get API version from header
    
    Args:
        version_header: API version header
    
    Returns:
        API version
    """
    if version_header:
        try:
            return APIVersion(version_header.lower())
        except ValueError:
            pass
    return APIVersion.LATEST


def create_versioned_router(version: APIVersion, prefix: str = "") -> APIRouter:
    """
    Create a versioned router
    
    Args:
        version: API version
        prefix: Additional prefix
    
    Returns:
        Versioned router
    """
    version_prefix = f"/{version.value}"
    if prefix:
        version_prefix = f"{version_prefix}{prefix}"
    
    return APIRouter(prefix=version_prefix)

