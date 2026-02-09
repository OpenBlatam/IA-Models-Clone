"""
Version Router
==============

API version routing.
"""

import logging
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class VersionRouter:
    """API version router."""
    
    def __init__(self, version_manager: Any):
        self.version_manager = version_manager
        self._routes: Dict[str, Dict[str, Callable]] = {}  # version -> path -> handler
    
    def register_route(
        self,
        version: str,
        path: str,
        handler: Callable
    ):
        """Register route for version."""
        if version not in self._routes:
            self._routes[version] = {}
        
        self._routes[version][path] = handler
        logger.info(f"Registered route {path} for version {version}")
    
    def get_handler(self, version: str, path: str) -> Optional[Callable]:
        """Get handler for version and path."""
        return self._routes.get(version, {}).get(path)
    
    def extract_version(self, request: Request) -> Optional[str]:
        """Extract version from request."""
        # Check header
        version = request.headers.get("API-Version")
        if version:
            return version
        
        # Check query parameter
        version = request.query_params.get("version")
        if version:
            return version
        
        # Check path
        path_parts = request.url.path.split("/")
        if len(path_parts) > 1 and path_parts[1].startswith("v"):
            return path_parts[1]
        
        # Return default
        return self.version_manager.get_default_version()
    
    async def route_request(self, request: Request, path: str) -> Optional[Response]:
        """Route request to appropriate version handler."""
        version = self.extract_version(request)
        
        if not version:
            version = self.version_manager.get_default_version()
        
        # Check version status
        api_version = self.version_manager.get_version(version)
        if not api_version:
            return None
        
        if api_version.status.value == "sunset":
            # Return 410 Gone
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=410,
                content={
                    "error": "API version has been sunset",
                    "version": version,
                    "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None
                }
            )
        
        # Get handler
        handler = self.get_handler(version, path)
        if not handler:
            return None
        
        # Add version header to response
        response = await handler(request)
        if hasattr(response, "headers"):
            response.headers["API-Version"] = version
            if api_version.status.value == "deprecated":
                response.headers["Deprecation"] = "true"
                if api_version.deprecation_date:
                    response.headers["Sunset"] = api_version.sunset_date.isoformat() if api_version.sunset_date else ""
        
        return response















