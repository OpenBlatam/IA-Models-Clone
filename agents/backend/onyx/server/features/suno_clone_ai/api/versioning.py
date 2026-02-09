"""
Sistema de versionado de API
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, Header, HTTPException, status
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class VersionedAPIRoute(APIRoute):
    """Route que soporta versionado"""
    
    def __init__(self, *args, version: Optional[str] = None, **kwargs):
        self.version = version
        super().__init__(*args, **kwargs)


def get_api_version(request: Request, api_version: Optional[str] = Header(None, alias="X-API-Version")) -> str:
    """
    Dependency para obtener la versión de API desde el header
    
    Args:
        request: Request object
        api_version: Versión de API desde header X-API-Version
    
    Returns:
        Versión de API (default: "v1")
    """
    if api_version:
        return api_version
    
    # Intentar obtener desde path
    path_parts = request.url.path.split("/")
    if len(path_parts) > 1 and path_parts[1].startswith("v"):
        return path_parts[1]
    
    return "v1"


def validate_api_version(version: str, supported_versions: list[str] = None) -> str:
    """
    Valida que la versión de API sea soportada
    
    Args:
        version: Versión a validar
        supported_versions: Lista de versiones soportadas
    
    Returns:
        Versión validada
    
    Raises:
        HTTPException si la versión no es soportada
    """
    if supported_versions is None:
        supported_versions = ["v1"]
    
    if version not in supported_versions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API version '{version}' not supported. Supported versions: {', '.join(supported_versions)}"
        )
    
    return version


def create_versioned_router(version: str, prefix: str = "", **kwargs) -> APIRouter:
    """
    Crea un router versionado
    
    Args:
        version: Versión de API (ej: "v1", "v2")
        prefix: Prefijo del router
        **kwargs: Argumentos adicionales para APIRouter
    
    Returns:
        APIRouter versionado
    """
    version_prefix = f"/{version}{prefix}" if prefix else f"/{version}"
    
    return APIRouter(
        prefix=version_prefix,
        tags=[f"API {version.upper()}"],
        **kwargs
    )


# Versiones soportadas
SUPPORTED_VERSIONS = ["v1"]

# Router para versiones
versions_router = APIRouter(prefix="/versions", tags=["versions"])


@versions_router.get("")
async def list_versions():
    """Lista las versiones de API disponibles"""
    return {
        "supported_versions": SUPPORTED_VERSIONS,
        "default_version": "v1",
        "deprecated_versions": [],
        "endpoints": {
            "v1": {
                "status": "stable",
                "base_path": "/v1"
            }
        }
    }

