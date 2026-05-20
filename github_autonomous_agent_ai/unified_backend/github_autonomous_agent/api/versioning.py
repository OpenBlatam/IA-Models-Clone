"""
Sistema de Versionado de API.
"""

from typing import Optional, Callable
from fastapi import Request, HTTPException
from fastapi.routing import APIRoute
from starlette.responses import Response

from config.logging_config import get_logger

logger = get_logger(__name__)


class APIVersion:
    """Versión de API."""
    
    def __init__(self, version: str, deprecated: bool = False, sunset_date: Optional[str] = None):
        """
        Inicializar versión de API.
        
        Args:
            version: Versión (ej: "v1", "v2")
            deprecated: Si está deprecada
            sunset_date: Fecha de sunset (opcional)
        """
        self.version = version
        self.deprecated = deprecated
        self.sunset_date = sunset_date


class VersionedRoute(APIRoute):
    """Ruta con versionado."""
    
    def __init__(
        self,
        path: str,
        endpoint: Callable,
        api_version: str = "v1",
        deprecated: bool = False,
        **kwargs
    ):
        """
        Inicializar ruta versionada.
        
        Args:
            path: Path de la ruta
            endpoint: Endpoint handler
            api_version: Versión de API
            deprecated: Si está deprecada
            **kwargs: Argumentos adicionales para APIRoute
        """
        # Agregar versión al path si no está presente
        if not path.startswith(f"/api/{api_version}"):
            path = f"/api/{api_version}{path}"
        
        super().__init__(path, endpoint, **kwargs)
        self.api_version = api_version
        self.deprecated = deprecated
    
    def get_route_handler(self):
        """Obtener handler de ruta con headers de versión."""
        original_route_handler = super().get_route_handler()
        
        async def versioned_route_handler(request: Request) -> Response:
            response = await original_route_handler(request)
            
            # Agregar headers de versión
            response.headers["API-Version"] = self.api_version
            if self.deprecated:
                response.headers["Deprecated"] = "true"
                response.headers["Warning"] = f'299 - "This API version is deprecated"'
            
            return response
        
        return versioned_route_handler


def get_api_version(request: Request) -> str:
    """
    Obtener versión de API del request.
    
    Args:
        request: Request de FastAPI
        
    Returns:
        Versión de API (ej: "v1")
    """
    # Intentar obtener de path
    path = request.url.path
    if "/api/v" in path:
        parts = path.split("/api/v")
        if len(parts) > 1:
            version_part = parts[1].split("/")[0]
            if version_part.isdigit():
                return f"v{version_part}"
    
    # Intentar obtener de header
    api_version = request.headers.get("API-Version", "v1")
    return api_version


def require_api_version(min_version: str = "v1", max_version: Optional[str] = None):
    """
    Decorador para requerir versión específica de API.
    
    Args:
        min_version: Versión mínima requerida
        max_version: Versión máxima permitida (opcional)
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            version = get_api_version(request)
            
            # Comparar versiones
            version_num = int(version.replace("v", ""))
            min_version_num = int(min_version.replace("v", ""))
            
            if version_num < min_version_num:
                raise HTTPException(
                    status_code=400,
                    detail=f"API version {version} is not supported. Minimum version: {min_version}"
                )
            
            if max_version:
                max_version_num = int(max_version.replace("v", ""))
                if version_num > max_version_num:
                    raise HTTPException(
                        status_code=400,
                        detail=f"API version {version} is not supported. Maximum version: {max_version}"
                    )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator



