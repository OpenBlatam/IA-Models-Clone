"""
API Versioning - Versionado de API
===================================

Sistema de versionado de API para mantener compatibilidad.
"""

from enum import Enum
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.routing import APIRoute


class APIVersion(str, Enum):
    """Versiones de API disponibles."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


def get_api_version(api_version: Optional[str] = Header(None, alias="API-Version")) -> APIVersion:
    """
    Obtener versión de API del header.
    
    Args:
        api_version: Versión de API del header.
    
    Returns:
        Versión de API.
    
    Raises:
        HTTPException: Si la versión no es válida.
    """
    if api_version is None:
        return APIVersion.LATEST
    
    try:
        return APIVersion(api_version.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API version. Supported: {[v.value for v in APIVersion]}"
        )


def create_versioned_router(version: APIVersion, prefix: str = "/api") -> APIRouter:
    """
    Crear router versionado.
    
    Args:
        version: Versión de API.
        prefix: Prefijo base.
    
    Returns:
        Router versionado.
    """
    return APIRouter(
        prefix=f"{prefix}/{version.value}",
        tags=[f"API {version.value.upper()}"]
    )


class VersionedRoute(APIRoute):
    """Route con información de versión."""
    
    def __init__(self, *args, version: APIVersion = APIVersion.LATEST, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version




