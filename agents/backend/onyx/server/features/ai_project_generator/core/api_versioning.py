"""
API Versioning - Versionado de APIs
===================================

Sistema de versionado de APIs:
- URL-based versioning
- Header-based versioning
- Query parameter versioning
- Version negotiation
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from fastapi import Request, HTTPException, status
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class VersioningStrategy(str, Enum):
    """Estrategias de versionado"""
    URL = "url"  # /api/v1/endpoint
    HEADER = "header"  # Accept: application/vnd.api.v1+json
    QUERY = "query"  # ?version=1
    SUBDOMAIN = "subdomain"  # v1.api.example.com


class APIVersionRouter:
    """
    Router de versionado de APIs - Extrae y rutea requests según versión.
    """
    
    def __init__(
        self,
        default_version: str = "v1",
        supported_versions: Optional[List[str]] = None,
        strategy: VersioningStrategy = VersioningStrategy.URL
    ) -> None:
        self.default_version = default_version
        self.supported_versions = supported_versions or [default_version]
        self.strategy = strategy
        self.version_handlers: Dict[str, Dict[str, Callable]] = {}
    
    def register_version(
        self,
        version: str,
        endpoint: str,
        handler: Callable
    ) -> None:
        """Registra handler para una versión"""
        if version not in self.version_handlers:
            self.version_handlers[version] = {}
        self.version_handlers[version][endpoint] = handler
        logger.info(f"Registered handler for {endpoint} at version {version}")
    
    def get_version(self, request: Request) -> str:
        """Obtiene versión del request"""
        if self.strategy == VersioningStrategy.URL:
            return self._get_version_from_url(request)
        elif self.strategy == VersioningStrategy.HEADER:
            return self._get_version_from_header(request)
        elif self.strategy == VersioningStrategy.QUERY:
            return self._get_version_from_query(request)
        else:
            return self.default_version
    
    def _get_version_from_url(self, request: Request) -> str:
        """Extrae versión de la URL"""
        path = request.url.path
        parts = path.split("/")
        
        for i, part in enumerate(parts):
            if part.startswith("v") and part[1:].isdigit():
                version = part
                if version in self.supported_versions:
                    return version
        
        return self.default_version
    
    def _get_version_from_header(self, request: Request) -> str:
        """Extrae versión del header Accept"""
        accept = request.headers.get("Accept", "")
        
        # Buscar formato: application/vnd.api.v1+json
        if "vnd.api.v" in accept:
            import re
            match = re.search(r"vnd\.api\.v(\d+)", accept)
            if match:
                version = f"v{match.group(1)}"
                if version in self.supported_versions:
                    return version
        
        # Buscar header X-API-Version
        api_version = request.headers.get("X-API-Version")
        if api_version and api_version in self.supported_versions:
            return api_version
        
        return self.default_version
    
    def _get_version_from_query(self, request: Request) -> str:
        """Extrae versión de query parameter"""
        version = request.query_params.get("version")
        if version:
            version_str = f"v{version}" if not version.startswith("v") else version
            if version_str in self.supported_versions:
                return version_str
        
        return self.default_version
    
    def get_handler(self, version: str, endpoint: str) -> Optional[Callable]:
        """Obtiene handler para versión y endpoint"""
        version_handlers = self.version_handlers.get(version, {})
        return version_handlers.get(endpoint)
    
    def is_version_supported(self, version: str) -> bool:
        """Verifica si versión está soportada"""
        return version in self.supported_versions


def create_versioned_route(
    path: str,
    version: str,
    handler: Callable,
    methods: Optional[List[str]] = None
) -> APIRoute:
    """Crea ruta versionada"""
    versioned_path = f"/api/{version}{path}"
    return APIRoute(
        versioned_path,
        handler,
        methods=methods or ["GET"]
    )


def get_api_version_router(
    default_version: str = "v1",
    supported_versions: Optional[List[str]] = None,
    strategy: VersioningStrategy = VersioningStrategy.URL
) -> APIVersionRouter:
    """Obtiene router de versionado"""
    return APIVersionRouter(
        default_version=default_version,
        supported_versions=supported_versions,
        strategy=strategy
    )


def get_api_version_manager(
    default_version: str = "v1",
    supported_versions: Optional[List[str]] = None,
    strategy: VersioningStrategy = VersioningStrategy.URL
) -> APIVersionRouter:
    """Obtiene gestor de versionado (alias para backward compatibility)"""
    return get_api_version_router(default_version, supported_versions, strategy)















