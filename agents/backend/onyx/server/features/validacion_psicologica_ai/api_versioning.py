"""
Sistema de Versionado de API
=============================
Versionado de endpoints y compatibilidad
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import structlog

logger = structlog.get_logger()


class APIVersion(str, Enum):
    """Versiones de API"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "v3"


class VersionInfo:
    """Información de versión"""
    
    def __init__(
        self,
        version: APIVersion,
        release_date: datetime,
        changes: List[str],
        deprecated: bool = False,
        deprecation_date: Optional[datetime] = None
    ):
        self.version = version
        self.release_date = release_date
        self.changes = changes
        self.deprecated = deprecated
        self.deprecation_date = deprecation_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "version": self.version.value,
            "release_date": self.release_date.isoformat(),
            "changes": self.changes,
            "deprecated": self.deprecated,
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None
        }


class APIVersionManager:
    """Gestor de versionado de API"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._versions: Dict[APIVersion, VersionInfo] = {}
        self._load_version_info()
        logger.info("APIVersionManager initialized")
    
    def _load_version_info(self) -> None:
        """Cargar información de versiones"""
        # Versión 1
        self._versions[APIVersion.V1] = VersionInfo(
            version=APIVersion.V1,
            release_date=datetime(2024, 1, 1),
            changes=[
                "Initial API release",
                "Basic validation endpoints",
                "Social media connections"
            ]
        )
        
        # Versión 2
        self._versions[APIVersion.V2] = VersionInfo(
            version=APIVersion.V2,
            release_date=datetime(2024, 6, 1),
            changes=[
                "Added batch processing",
                "Enhanced analytics",
                "GraphQL support"
            ]
        )
        
        # Versión 3 (Latest)
        self._versions[APIVersion.V3] = VersionInfo(
            version=APIVersion.V3,
            release_date=datetime(2024, 12, 1),
            changes=[
                "AI integrations",
                "Event bus",
                "Advanced metrics",
                "A/B testing"
            ]
        )
    
    def get_version_info(self, version: APIVersion) -> Optional[VersionInfo]:
        """
        Obtener información de versión
        
        Args:
            version: Versión
            
        Returns:
            Información de versión o None
        """
        return self._versions.get(version)
    
    def get_all_versions(self) -> List[VersionInfo]:
        """
        Obtener todas las versiones
        
        Returns:
            Lista de versiones
        """
        return list(self._versions.values())
    
    def get_latest_version(self) -> VersionInfo:
        """
        Obtener última versión
        
        Returns:
            Información de última versión
        """
        return self._versions[APIVersion.LATEST]
    
    def is_version_deprecated(self, version: APIVersion) -> bool:
        """
        Verificar si versión está deprecada
        
        Args:
            version: Versión
            
        Returns:
            True si está deprecada
        """
        version_info = self._versions.get(version)
        return version_info.deprecated if version_info else False
    
    def get_version_changes(self, from_version: APIVersion, to_version: APIVersion) -> List[str]:
        """
        Obtener cambios entre versiones
        
        Args:
            from_version: Versión origen
            to_version: Versión destino
            
        Returns:
            Lista de cambios
        """
        from_info = self._versions.get(from_version)
        to_info = self._versions.get(to_version)
        
        if not from_info or not to_info:
            return []
        
        # Simplificado: retornar cambios de la versión destino
        return to_info.changes


# Instancia global del gestor de versionado
api_version_manager = APIVersionManager()




