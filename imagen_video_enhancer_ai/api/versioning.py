"""
API Versioning
==============

Utilities for API versioning.
"""

from typing import Optional, Callable, Dict
from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
import re


class VersionedRouter:
    """Router with versioning support."""
    
    def __init__(self, default_version: str = "v1"):
        """
        Initialize versioned router.
        
        Args:
            default_version: Default API version
        """
        self.default_version = default_version
        self.routers: Dict[str, APIRouter] = {}
    
    def get_router(self, version: Optional[str] = None) -> APIRouter:
        """
        Get router for version.
        
        Args:
            version: API version (defaults to default_version)
            
        Returns:
            APIRouter for the version
        """
        version = version or self.default_version
        
        if version not in self.routers:
            self.routers[version] = APIRouter(
                prefix=f"/api/{version}",
                tags=[f"API {version}"]
            )
        
        return self.routers[version]
    
    def register_version(self, version: str, router: APIRouter):
        """
        Register a router for a version.
        
        Args:
            version: API version
            router: Router to register
        """
        self.routers[version] = router


def extract_version(request: Request) -> Optional[str]:
    """
    Extract API version from request.
    
    Checks:
    - URL path (/api/v1/...)
    - Header (X-API-Version)
    - Query parameter (version)
    
    Args:
        request: FastAPI request
        
    Returns:
        API version or None
    """
    # Check URL path
    path_match = re.search(r'/api/(v\d+)/', request.url.path)
    if path_match:
        return path_match.group(1)
    
    # Check header
    version_header = request.headers.get("X-API-Version")
    if version_header:
        return version_header
    
    # Check query parameter
    version_param = request.query_params.get("version")
    if version_param:
        return version_param
    
    return None


def version_route(version: str):
    """
    Decorator to mark route with version.
    
    Args:
        version: API version
        
    Usage:
        @version_route("v1")
        @router.get("/endpoint")
        async def my_endpoint():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._api_version = version
        return func
    
    return decorator

