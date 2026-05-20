"""
MCP API Versioning - Versionado de APIs
========================================
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum
from fastapi import APIRouter, Request
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """Versiones de API soportadas"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class VersionedRouter:
    """
    Router con soporte para versionado de APIs
    
    Permite mantener múltiples versiones de endpoints simultáneamente.
    """
    
    def __init__(self, default_version: APIVersion = APIVersion.V1):
        """
        Args:
            default_version: Versión por defecto
        """
        self.default_version = default_version
        self._routers: Dict[APIVersion, APIRouter] = {
            version: APIRouter() for version in APIVersion
        }
        self._version_handlers: Dict[APIVersion, Dict[str, callable]] = {
            version: {} for version in APIVersion
        }
    
    def add_endpoint(
        self,
        path: str,
        endpoint: callable,
        methods: list[str],
        version: Optional[APIVersion] = None,
    ):
        """
        Agrega endpoint a una versión específica
        
        Args:
            path: Ruta del endpoint
            endpoint: Función del endpoint
            methods: Métodos HTTP
            version: Versión (usa default si no se especifica)
        """
        version = version or self.default_version
        router = self._routers[version]
        
        for method in methods:
            router.add_api_route(
                path,
                endpoint,
                methods=[method.upper()],
            )
        
        # Registrar handler
        key = f"{method}:{path}"
        if key not in self._version_handlers[version]:
            self._version_handlers[version][key] = endpoint
        
        logger.info(f"Added endpoint {path} to version {version.value}")
    
    def get_router(self, version: Optional[APIVersion] = None) -> APIRouter:
        """
        Obtiene router para una versión
        
        Args:
            version: Versión (retorna router combinado si None)
            
        Returns:
            APIRouter
        """
        if version:
            return self._routers[version]
        
        # Combinar todos los routers
        combined = APIRouter()
        for v in APIVersion:
            combined.include_router(
                self._routers[v],
                prefix=f"/{v.value}",
            )
        return combined
    
    def get_version_from_request(self, request: Request) -> APIVersion:
        """
        Extrae versión de API desde request
        
        Args:
            request: Request de FastAPI
            
        Returns:
            Versión de API
        """
        # Intentar desde header
        api_version = request.headers.get("X-API-Version")
        if api_version:
            try:
                return APIVersion(api_version.lower())
            except ValueError:
                pass
        
        # Intentar desde path
        path = request.url.path
        for version in APIVersion:
            if f"/{version.value}/" in path:
                return version
        
        # Usar default
        return self.default_version


def versioned_route(version: APIVersion):
    """
    Decorador para marcar endpoints con versión
    
    Args:
        version: Versión de la API
    """
    def decorator(func):
        func._api_version = version
        return func
    return decorator

