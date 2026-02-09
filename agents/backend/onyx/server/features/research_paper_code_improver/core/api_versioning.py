"""
API Versioning - Sistema de versionado de APIs
================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """Estrategias de versionado"""
    URL_PATH = "url_path"  # /api/v1/endpoint
    HEADER = "header"  # X-API-Version: v1
    QUERY_PARAM = "query_param"  # ?version=v1
    ACCEPT_HEADER = "accept_header"  # Accept: application/vnd.api.v1+json


@dataclass
class APIVersion:
    """Versión de API"""
    version: str  # "v1", "v2", etc.
    released_at: datetime
    deprecated_at: Optional[datetime] = None
    sunset_at: Optional[datetime] = None
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_deprecated(self) -> bool:
        """Verifica si está deprecada"""
        if self.deprecated_at is None:
            return False
        return datetime.now() >= self.deprecated_at
    
    def is_sunset(self) -> bool:
        """Verifica si está en sunset"""
        if self.sunset_at is None:
            return False
        return datetime.now() >= self.sunset_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "version": self.version,
            "released_at": self.released_at.isoformat(),
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "sunset_at": self.sunset_at.isoformat() if self.sunset_at else None,
            "is_deprecated": self.is_deprecated(),
            "is_sunset": self.is_sunset(),
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
            "metadata": self.metadata
        }


@dataclass
class VersionedEndpoint:
    """Endpoint versionado"""
    path: str
    method: str
    handler: Callable
    versions: List[str]  # Versiones que soporta
    default_version: str = "v1"


class APIVersioning:
    """Sistema de versionado de APIs"""
    
    def __init__(self, strategy: VersionStrategy = VersionStrategy.URL_PATH):
        self.strategy = strategy
        self.versions: Dict[str, APIVersion] = {}
        self.endpoints: List[VersionedEndpoint] = []
        self.default_version = "v1"
    
    def register_version(
        self,
        version: str,
        released_at: datetime,
        deprecated_at: Optional[datetime] = None,
        sunset_at: Optional[datetime] = None,
        changelog: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIVersion:
        """Registra una versión de API"""
        api_version = APIVersion(
            version=version,
            released_at=released_at,
            deprecated_at=deprecated_at,
            sunset_at=sunset_at,
            changelog=changelog or [],
            breaking_changes=breaking_changes or [],
            metadata=metadata or {}
        )
        
        self.versions[version] = api_version
        logger.info(f"Versión {version} registrada")
        return api_version
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        handler: Callable,
        versions: List[str],
        default_version: Optional[str] = None
    ) -> VersionedEndpoint:
        """Registra un endpoint versionado"""
        endpoint = VersionedEndpoint(
            path=path,
            method=method,
            handler=handler,
            versions=versions,
            default_version=default_version or self.default_version
        )
        
        self.endpoints.append(endpoint)
        return endpoint
    
    def extract_version(
        self,
        path: str,
        headers: Dict[str, str],
        query_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Extrae la versión de una petición"""
        if self.strategy == VersionStrategy.URL_PATH:
            # Extraer de path: /api/v1/endpoint
            parts = path.split("/")
            for i, part in enumerate(parts):
                if part.startswith("v") and part[1:].isdigit():
                    return part
            return self.default_version
        
        elif self.strategy == VersionStrategy.HEADER:
            # Extraer de header: X-API-Version
            version = headers.get("X-API-Version") or headers.get("x-api-version")
            if version:
                return version
            return self.default_version
        
        elif self.strategy == VersionStrategy.QUERY_PARAM:
            # Extraer de query param: ?version=v1
            if query_params:
                version = query_params.get("version")
                if version:
                    return version
            return self.default_version
        
        elif self.strategy == VersionStrategy.ACCEPT_HEADER:
            # Extraer de Accept header: application/vnd.api.v1+json
            accept = headers.get("Accept") or headers.get("accept", "")
            import re
            match = re.search(r'vnd\.api\.(v\d+)', accept)
            if match:
                return match.group(1)
            return self.default_version
        
        return self.default_version
    
    def get_endpoint_handler(
        self,
        path: str,
        method: str,
        version: Optional[str] = None
    ) -> Optional[Callable]:
        """Obtiene el handler para un endpoint y versión"""
        if version is None:
            version = self.default_version
        
        for endpoint in self.endpoints:
            if endpoint.method.upper() == method.upper() and self._path_matches(endpoint.path, path):
                if version in endpoint.versions:
                    return endpoint.handler
                elif endpoint.default_version in endpoint.versions:
                    return endpoint.handler
        
        return None
    
    def _path_matches(self, pattern: str, path: str) -> bool:
        """Verifica si un path coincide con un patrón"""
        # Simplificado - en producción usar routing más sofisticado
        return pattern == path or path.startswith(pattern)
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de una versión"""
        if version not in self.versions:
            return None
        return self.versions[version].to_dict()
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """Lista todas las versiones"""
        return [v.to_dict() for v in self.versions.values()]
    
    def get_latest_version(self) -> Optional[str]:
        """Obtiene la versión más reciente"""
        if not self.versions:
            return None
        
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].released_at,
            reverse=True
        )
        return sorted_versions[0][0]
    
    def deprecate_version(self, version: str, deprecated_at: datetime):
        """Marca una versión como deprecada"""
        if version in self.versions:
            self.versions[version].deprecated_at = deprecated_at
            logger.warning(f"Versión {version} marcada como deprecada")
    
    def sunset_version(self, version: str, sunset_at: datetime):
        """Marca una versión para sunset"""
        if version in self.versions:
            self.versions[version].sunset_at = sunset_at
            logger.warning(f"Versión {version} programada para sunset")




