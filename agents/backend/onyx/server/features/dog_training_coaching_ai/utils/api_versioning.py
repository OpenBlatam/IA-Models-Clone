"""
API Versioning
==============
Utilidades para versionado de API.
"""

from typing import Optional, Callable, Dict, Any
from fastapi import Request, HTTPException
from enum import Enum
import re


class APIVersion(str, Enum):
    """Versiones de API."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class VersionManager:
    """Manager para versionado de API."""
    
    def __init__(self, default_version: APIVersion = APIVersion.V1):
        self.default_version = default_version
        self.versions: Dict[APIVersion, Dict[str, Any]] = {}
        self.deprecated_versions: set = set()
    
    def register_version(
        self,
        version: APIVersion,
        description: str = "",
        deprecated: bool = False
    ):
        """
        Registrar una versión de API.
        
        Args:
            version: Versión a registrar
            description: Descripción de la versión
            deprecated: Si está deprecada
        """
        self.versions[version] = {
            "description": description,
            "deprecated": deprecated,
            "endpoints": []
        }
        
        if deprecated:
            self.deprecated_versions.add(version)
    
    def get_version_from_request(self, request: Request) -> APIVersion:
        """
        Obtener versión desde request.
        
        Args:
            request: Request de FastAPI
            
        Returns:
            Versión de API
        """
        # Intentar obtener de path
        path = request.url.path
        version_match = re.search(r'/api/(v\d+)/', path)
        if version_match:
            version_str = version_match.group(1)
            try:
                return APIVersion(version_str)
            except ValueError:
                pass
        
        # Intentar obtener de header
        version_header = request.headers.get("X-API-Version")
        if version_header:
            try:
                return APIVersion(version_header)
            except ValueError:
                pass
        
        # Usar versión por defecto
        return self.default_version
    
    def check_deprecated(self, version: APIVersion) -> Optional[str]:
        """
        Verificar si una versión está deprecada.
        
        Args:
            version: Versión a verificar
            
        Returns:
            Mensaje de deprecación o None
        """
        if version in self.deprecated_versions:
            return f"API version {version.value} is deprecated. Please upgrade to a newer version."
        return None


def require_version(required_version: APIVersion):
    """
    Decorador para requerir versión específica.
    
    Args:
        required_version: Versión requerida
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Buscar Request en kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if request:
                from ...utils.api_versioning import VersionManager
                manager = VersionManager()
                version = manager.get_version_from_request(request)
                
                if version != required_version:
                    raise HTTPException(
                        status_code=400,
                        detail=f"This endpoint requires API version {required_version.value}, but {version.value} was requested"
                    )
                
                # Verificar deprecación
                deprecation_msg = manager.check_deprecated(version)
                if deprecation_msg:
                    # Agregar header de advertencia
                    pass  # Se puede agregar header aquí
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

