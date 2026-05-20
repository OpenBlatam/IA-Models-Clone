"""
API Version Manager - Gestor de Versiones de API
================================================

Gestión avanzada de versiones de API:
- Version negotiation
- Version deprecation
- Migration guides
- Version compatibility
- Sunset policies
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """Estados de versión"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"


class APIVersion:
    """Versión de API"""
    
    def __init__(
        self,
        version: str,
        status: VersionStatus = VersionStatus.ACTIVE,
        release_date: Optional[datetime] = None,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None
    ) -> None:
        self.version = version
        self.status = status
        self.release_date = release_date or datetime.now()
        self.deprecation_date = deprecation_date
        self.sunset_date = sunset_date
        self.migration_guide: Optional[str] = None
        self.changelog: List[str] = []
    
    def is_active(self) -> bool:
        """Verifica si está activa"""
        return self.status == VersionStatus.ACTIVE
    
    def is_deprecated(self) -> bool:
        """Verifica si está deprecada"""
        return self.status == VersionStatus.DEPRECATED
    
    def should_sunset(self) -> bool:
        """Verifica si debe hacer sunset"""
        if self.sunset_date:
            return datetime.now() >= self.sunset_date
        return False


class APIVersionManager:
    """
    Gestor de versiones de API.
    """
    
    def __init__(self) -> None:
        self.versions: Dict[str, APIVersion] = {}
        self.default_version = "v1"
        self.version_strategies: Dict[str, str] = {}  # endpoint -> strategy
    
    def register_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.ACTIVE,
        release_date: Optional[datetime] = None
    ) -> APIVersion:
        """Registra versión"""
        api_version = APIVersion(version, status, release_date)
        self.versions[version] = api_version
        logger.info(f"API version registered: {version}")
        return api_version
    
    def deprecate_version(
        self,
        version: str,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None,
        migration_guide: Optional[str] = None
    ) -> None:
        """Depreca versión"""
        if version in self.versions:
            api_version = self.versions[version]
            api_version.status = VersionStatus.DEPRECATED
            api_version.deprecation_date = deprecation_date or datetime.now()
            api_version.sunset_date = sunset_date
            api_version.migration_guide = migration_guide
            logger.warning(f"API version deprecated: {version}")
    
    def get_version(self, version: Optional[str] = None) -> Optional[APIVersion]:
        """Obtiene versión"""
        if not version:
            version = self.default_version
        return self.versions.get(version)
    
    def negotiate_version(
        self,
        requested_version: Optional[str] = None,
        accept_header: Optional[str] = None
    ) -> Optional[str]:
        """Negocia versión"""
        # Si se especifica versión, usarla
        if requested_version and requested_version in self.versions:
            api_version = self.versions[requested_version]
            if api_version.is_active() and not api_version.should_sunset():
                return requested_version
        
        # Parsear Accept header
        if accept_header:
            # Formato: application/vnd.api+json;version=v2
            if "version=" in accept_header:
                version = accept_header.split("version=")[1].split(";")[0]
                if version in self.versions:
                    return version
        
        # Usar versión por defecto
        return self.default_version
    
    def get_version_info(self, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Obtiene información de versión"""
        api_version = self.get_version(version)
        if not api_version:
            return None
        
        return {
            "version": api_version.version,
            "status": api_version.status.value,
            "release_date": api_version.release_date.isoformat(),
            "deprecation_date": api_version.deprecation_date.isoformat() if api_version.deprecation_date else None,
            "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None,
            "migration_guide": api_version.migration_guide,
            "changelog": api_version.changelog
        }
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """Obtiene todas las versiones"""
        return [
            self.get_version_info(version)
            for version in sorted(self.versions.keys(), reverse=True)
        ]
    
    def add_changelog_entry(self, version: str, entry: str) -> None:
        """Agrega entrada a changelog"""
        if version in self.versions:
            self.versions[version].changelog.append(entry)


def get_api_version_manager() -> APIVersionManager:
    """Obtiene gestor de versiones de API"""
    return APIVersionManager()















