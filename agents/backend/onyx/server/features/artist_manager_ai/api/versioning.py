"""
API Versioning
==============

Sistema de versionado de API.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum
from fastapi import APIRouter, Request
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Versiones de API."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "latest"


class VersionedAPIRouter(APIRouter):
    """Router con versionado."""
    
    def __init__(self, version: APIVersion = APIVersion.V1, *args, **kwargs):
        """
        Inicializar router versionado.
        
        Args:
            version: Versión de la API
            *args: Argumentos para APIRouter
            **kwargs: Argumentos para APIRouter
        """
        prefix = kwargs.get("prefix", "")
        if prefix and not prefix.startswith(f"/{version.value}"):
            kwargs["prefix"] = f"/{version.value}{prefix}"
        elif not prefix:
            kwargs["prefix"] = f"/{version.value}"
        
        kwargs["tags"] = kwargs.get("tags", []) + [f"API {version.value.upper()}"]
        
        super().__init__(*args, **kwargs)
        self.version = version
        self._logger = logger


def get_api_version(request: Request) -> APIVersion:
    """
    Obtener versión de API desde request.
    
    Args:
        request: Request de FastAPI
    
    Returns:
        Versión de API
    """
    # Intentar obtener de header
    version_header = request.headers.get("API-Version", "v1")
    
    # Intentar obtener de path
    path = request.url.path
    if "/v1/" in path:
        return APIVersion.V1
    elif "/v2/" in path:
        return APIVersion.V2
    
    # Mapear header a enum
    version_map = {
        "v1": APIVersion.V1,
        "v2": APIVersion.V2,
        "latest": APIVersion.LATEST
    }
    
    return version_map.get(version_header.lower(), APIVersion.V1)




