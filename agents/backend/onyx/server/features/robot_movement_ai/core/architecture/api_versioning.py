"""
Sistema de versionado de API para Robot Movement AI v2.0
Manejo de múltiples versiones de API simultáneamente
"""

from typing import Optional, Callable, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from functools import wraps

try:
    from fastapi import APIRouter, Request
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class APIVersion(str, Enum):
    """Versiones de API disponibles"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


@dataclass
class VersionInfo:
    """Información de versión de API"""
    version: APIVersion
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changelog: str = ""


class APIVersionManager:
    """Gestor de versiones de API"""
    
    def __init__(self):
        """Inicializar gestor"""
        self.versions: Dict[APIVersion, VersionInfo] = {}
        self.default_version = APIVersion.V2
        self._register_default_versions()
    
    def _register_default_versions(self):
        """Registrar versiones por defecto"""
        self.versions[APIVersion.V1] = VersionInfo(
            version=APIVersion.V1,
            deprecated=True,
            deprecation_date=datetime(2024, 1, 1),
            changelog="Initial version"
        )
        
        self.versions[APIVersion.V2] = VersionInfo(
            version=APIVersion.V2,
            deprecated=False,
            changelog="Clean Architecture + DDD implementation"
        )
    
    def register_version(self, version_info: VersionInfo):
        """Registrar información de versión"""
        self.versions[version_info.version] = version_info
    
    def get_version_info(self, version: APIVersion) -> Optional[VersionInfo]:
        """Obtener información de versión"""
        return self.versions.get(version)
    
    def is_deprecated(self, version: APIVersion) -> bool:
        """Verificar si versión está deprecada"""
        info = self.get_version_info(version)
        return info.deprecated if info else False
    
    def get_latest_version(self) -> APIVersion:
        """Obtener última versión"""
        non_deprecated = [v for v, info in self.versions.items() if not info.deprecated]
        if non_deprecated:
            return max(non_deprecated, key=lambda x: x.value)
        return self.default_version


# Instancia global
_version_manager: Optional[APIVersionManager] = None


def get_version_manager() -> APIVersionManager:
    """Obtener instancia global del gestor de versiones"""
    global _version_manager
    if _version_manager is None:
        _version_manager = APIVersionManager()
    return _version_manager


def version_router(version: APIVersion, prefix: str = "/api"):
    """
    Crear router para versión específica
    
    Args:
        version: Versión de API
        prefix: Prefijo de ruta
        
    Returns:
        APIRouter configurado
    """
    if not FASTAPI_AVAILABLE:
        return None
    
    router = APIRouter(prefix=f"{prefix}/{version.value}")
    
    # Agregar middleware de deprecación
    @router.middleware("http")
    async def deprecation_middleware(request: Request, call_next):
        manager = get_version_manager()
        if manager.is_deprecated(version):
            response = await call_next(request)
            response.headers["X-API-Deprecated"] = "true"
            version_info = manager.get_version_info(version)
            if version_info and version_info.sunset_date:
                response.headers["X-API-Sunset"] = version_info.sunset_date.isoformat()
            return response
        
        return await call_next(request)
    
    return router


def require_version(version: APIVersion):
    """
    Decorator para requerir versión específica
    
    Usage:
        @require_version(APIVersion.V2)
        async def endpoint():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Verificar versión en request
            # Implementación simplificada
            return await func(*args, **kwargs)
        return wrapper
    return decorator




