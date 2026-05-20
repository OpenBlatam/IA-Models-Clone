"""
API Versioning System
Supports multiple API versions with backward compatibility
"""

from fastapi import APIRouter, Request
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """API versions"""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"  # Current latest version


class VersionRouter:
    """
    Router manager for API versioning.
    Supports multiple versions with different implementations.
    """
    
    def __init__(self):
        self.routers: Dict[str, APIRouter] = {}
        self.version_handlers: Dict[str, Dict[str, Any]] = {}
    
    def register_version(
        self,
        version: str,
        router: APIRouter,
        prefix: str = None,
        deprecated: bool = False,
        sunset_date: Optional[str] = None
    ):
        """
        Register router for API version
        
        Args:
            version: Version string (e.g., "v1", "v2")
            router: FastAPI router
            prefix: URL prefix (default: /api/{version})
            deprecated: Whether version is deprecated
            sunset_date: Date when version will be removed
        """
        prefix = prefix or f"/api/{version}"
        router.prefix = prefix
        
        self.routers[version] = router
        self.version_handlers[version] = {
            "router": router,
            "prefix": prefix,
            "deprecated": deprecated,
            "sunset_date": sunset_date
        }
        
        logger.info(f"Registered API version {version} with prefix {prefix}")
    
    def get_router(self, version: str) -> Optional[APIRouter]:
        """Get router for version"""
        return self.routers.get(version)
    
    def get_latest_router(self) -> Optional[APIRouter]:
        """Get latest version router"""
        return self.routers.get(APIVersion.LATEST.value)
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get information about all versions"""
        return {
            version: {
                "prefix": info["prefix"],
                "deprecated": info["deprecated"],
                "sunset_date": info["sunset_date"]
            }
            for version, info in self.version_handlers.items()
        }


def extract_api_version(request: Request) -> str:
    """
    Extract API version from request
    
    Checks:
    1. URL path (/api/v1/...)
    2. Header (X-API-Version)
    3. Query parameter (?version=v1)
    4. Accept header (application/vnd.api+json;version=1)
    
    Returns:
        Version string or latest version
    """
    # Check URL path
    path = request.url.path
    if "/api/v" in path:
        parts = path.split("/api/")
        if len(parts) > 1:
            version_part = parts[1].split("/")[0]
            if version_part.startswith("v"):
                return version_part
    
    # Check header
    version_header = request.headers.get("X-API-Version")
    if version_header:
        return version_header
    
    # Check query parameter
    version_param = request.query_params.get("version")
    if version_param:
        return version_param if version_param.startswith("v") else f"v{version_param}"
    
    # Check Accept header
    accept_header = request.headers.get("Accept", "")
    if "version=" in accept_header:
        # Parse version from Accept header
        for part in accept_header.split(";"):
            if "version=" in part:
                version_value = part.split("version=")[1].strip()
                return version_value if version_value.startswith("v") else f"v{version_value}"
    
    # Default to latest
    return APIVersion.LATEST.value


# Global version router
_version_router: Optional[VersionRouter] = None


def get_version_router() -> VersionRouter:
    """Get or create global version router"""
    global _version_router
    if _version_router is None:
        _version_router = VersionRouter()
    return _version_router















