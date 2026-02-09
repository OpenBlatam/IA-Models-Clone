"""
API Versioning - Sistema de Versionado de API
==============================================

Sistema completo de versionado de API.
"""

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Versiones de API."""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"
    LATEST = "latest"


@dataclass
class APIVersionInfo:
    """Información de versión de API."""
    version: str
    release_date: datetime
    status: str = "stable"  # stable, beta, deprecated
    changelog: List[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    migration_guide: Optional[str] = None


class APIVersionManager:
    """Gestor de versionado de API."""
    
    def __init__(self):
        self.versions: Dict[str, APIVersionInfo] = {}
        self.current_version = "v1"
        self.default_version = "v1"
        self._load_default_versions()
    
    def _load_default_versions(self):
        """Cargar versiones por defecto."""
        v1 = APIVersionInfo(
            version="v1",
            release_date=datetime.now(),
            status="stable",
            changelog=[
                "Versión inicial de la API",
                "Soporte completo para chat continuo",
                "Métricas y monitoreo",
            ],
        )
        
        self.versions["v1"] = v1
        self.current_version = "v1"
    
    def register_version(self, version_info: APIVersionInfo):
        """Registrar nueva versión."""
        self.versions[version_info.version] = version_info
        logger.info(f"Registered API version: {version_info.version}")
    
    def get_version_info(self, version: str) -> Optional[APIVersionInfo]:
        """Obtener información de versión."""
        # Resolver alias
        if version == "latest":
            version = self.current_version
        
        return self.versions.get(version)
    
    def is_version_supported(self, version: str) -> bool:
        """Verificar si una versión está soportada."""
        if version == "latest":
            return True
        
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        return not version_info.deprecated
    
    def deprecate_version(self, version: str, deprecation_date: Optional[datetime] = None):
        """Deprecar una versión."""
        version_info = self.get_version_info(version)
        if version_info:
            version_info.deprecated = True
            version_info.deprecation_date = deprecation_date or datetime.now()
            version_info.status = "deprecated"
            logger.warning(f"Deprecated API version: {version}")
    
    def get_supported_versions(self) -> List[str]:
        """Obtener lista de versiones soportadas."""
        return [
            v.version
            for v in self.versions.values()
            if not v.deprecated
        ]
    
    def get_all_versions(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las versiones."""
        return {
            version: {
                "version": info.version,
                "status": info.status,
                "deprecated": info.deprecated,
                "release_date": info.release_date.isoformat(),
                "changelog": info.changelog,
            }
            for version, info in self.versions.items()
        }
































