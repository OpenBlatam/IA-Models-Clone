"""
API Versioning - Sistema de versionado de API
=============================================

Sistema para manejar múltiples versiones de API con compatibilidad
hacia atrás y migración gradual.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Versiones de API disponibles"""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"  # Versión más reciente


@dataclass
class VersionInfo:
    """Información de una versión de API"""
    version: str
    released_at: datetime
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None
    sunset_at: Optional[datetime] = None
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)


class APIVersionManager:
    """
    Gestor de versiones de API.
    
    Permite manejar múltiples versiones de API con:
    - Compatibilidad hacia atrás
    - Deprecación gradual
    - Migración de versiones
    - Documentación de cambios
    """
    
    def __init__(self):
        self.versions: Dict[str, VersionInfo] = {}
        self.default_version = APIVersion.V1.value
        self.current_version = APIVersion.LATEST.value
        self.version_handlers: Dict[str, Dict[str, Callable]] = {}  # version -> endpoint -> handler
    
    def register_version(
        self,
        version: str,
        released_at: datetime,
        changelog: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None
    ) -> None:
        """
        Registrar una versión de API.
        
        Args:
            version: Versión (ej: "v1", "v2")
            released_at: Fecha de lanzamiento
            changelog: Lista de cambios
            breaking_changes: Lista de cambios que rompen compatibilidad
        """
        version_info = VersionInfo(
            version=version,
            released_at=released_at,
            changelog=changelog or [],
            breaking_changes=breaking_changes or []
        )
        self.versions[version] = version_info
        self.version_handlers[version] = {}
        logger.info(f"📦 API version {version} registered")
    
    def deprecate_version(
        self,
        version: str,
        deprecated_at: Optional[datetime] = None,
        sunset_at: Optional[datetime] = None
    ) -> None:
        """
        Marcar una versión como deprecada.
        
        Args:
            version: Versión a deprecar
            deprecated_at: Fecha de deprecación
            sunset_at: Fecha de eliminación (sunset)
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not registered")
        
        self.versions[version].deprecated = True
        self.versions[version].deprecated_at = deprecated_at or datetime.now()
        self.versions[version].sunset_at = sunset_at
        
        logger.warning(f"⚠️ API version {version} deprecated")
    
    def register_handler(
        self,
        version: str,
        endpoint: str,
        handler: Callable
    ) -> None:
        """
        Registrar handler para un endpoint en una versión específica.
        
        Args:
            version: Versión de API
            endpoint: Nombre del endpoint
            handler: Función handler
        """
        if version not in self.version_handlers:
            self.version_handlers[version] = {}
        
        self.version_handlers[version][endpoint] = handler
    
    def get_handler(self, version: str, endpoint: str) -> Optional[Callable]:
        """
        Obtener handler para un endpoint en una versión específica.
        
        Args:
            version: Versión de API
            endpoint: Nombre del endpoint
            
        Returns:
            Handler o None si no existe
        """
        # Intentar versión específica
        if version in self.version_handlers:
            if endpoint in self.version_handlers[version]:
                return self.version_handlers[version][endpoint]
        
        # Fallback a versión por defecto
        if self.default_version in self.version_handlers:
            if endpoint in self.version_handlers[self.default_version]:
                return self.version_handlers[self.default_version][endpoint]
        
        return None
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Obtener información de una versión"""
        return self.versions.get(version)
    
    def is_deprecated(self, version: str) -> bool:
        """Verificar si una versión está deprecada"""
        version_info = self.versions.get(version)
        return version_info.deprecated if version_info else False
    
    def is_sunset(self, version: str) -> bool:
        """Verificar si una versión ha sido eliminada (sunset)"""
        version_info = self.versions.get(version)
        if not version_info or not version_info.sunset_at:
            return False
        return datetime.now() >= version_info.sunset_at
    
    def get_supported_versions(self) -> List[str]:
        """Obtener lista de versiones soportadas"""
        return [
            v for v in self.versions.keys()
            if not self.is_sunset(v)
        ]
    
    def migrate_request(
        self,
        request_data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """
        Migrar request de una versión a otra.
        
        Args:
            request_data: Datos del request
            from_version: Versión origen
            to_version: Versión destino
            
        Returns:
            Datos migrados
        """
        # Por ahora, retornar datos sin cambios
        # En el futuro, se pueden agregar transformaciones específicas
        logger.debug(f"🔄 Migrating request from {from_version} to {to_version}")
        return request_data
    
    def get_version_headers(self, version: str) -> Dict[str, str]:
        """
        Obtener headers HTTP para una versión.
        
        Args:
            version: Versión de API
            
        Returns:
            Diccionario con headers
        """
        headers = {
            "X-API-Version": version,
            "X-API-Current-Version": self.current_version
        }
        
        version_info = self.versions.get(version)
        if version_info:
            if version_info.deprecated:
                headers["X-API-Deprecated"] = "true"
                if version_info.deprecated_at:
                    headers["X-API-Deprecated-At"] = version_info.deprecated_at.isoformat()
                if version_info.sunset_at:
                    headers["X-API-Sunset-At"] = version_info.sunset_at.isoformat()
        
        return headers


def extract_api_version(request_headers: Dict[str, str], default: str = "v1") -> str:
    """
    Extraer versión de API desde headers de request.
    
    Args:
        request_headers: Headers del request
        default: Versión por defecto
        
    Returns:
        Versión de API
    """
    # Buscar en diferentes headers comunes
    version = (
        request_headers.get("X-API-Version") or
        request_headers.get("API-Version") or
        request_headers.get("Accept", "").split("version=")[-1].split(";")[0] if "version=" in request_headers.get("Accept", "") else None
    )
    
    if version:
        # Limpiar versión (remover "v" si está presente)
        version = version.strip().lower()
        if version.startswith("v"):
            version = version[1:]
        return f"v{version}"
    
    return default




