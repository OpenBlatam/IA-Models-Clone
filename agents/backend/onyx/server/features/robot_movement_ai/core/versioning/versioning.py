"""
Versioning System
================

Sistema de versionado y gestión de versiones.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """Versión del sistema."""
    major: int
    minor: int
    patch: int
    build: Optional[str] = None
    prerelease: Optional[str] = None
    metadata: Optional[str] = None
    
    def __str__(self) -> str:
        """Representación de string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"+{self.build}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.metadata:
            version += f" ({self.metadata})"
        return version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)
    
    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """
        Crear versión desde string.
        
        Args:
            version_str: String de versión (ej: "1.2.3+build1-alpha")
            
        Returns:
            Version
        """
        # Parsear versión
        parts = version_str.split("+")
        base = parts[0]
        build = parts[1] if len(parts) > 1 else None
        
        parts = base.split("-")
        base = parts[0]
        prerelease = parts[1] if len(parts) > 1 else None
        
        major, minor, patch = map(int, base.split("."))
        
        return cls(
            major=major,
            minor=minor,
            patch=patch,
            build=build,
            prerelease=prerelease
        )
    
    def compare(self, other: "Version") -> int:
        """
        Comparar versiones.
        
        Args:
            other: Otra versión
            
        Returns:
            -1 si self < other, 0 si igual, 1 si self > other
        """
        if self.major != other.major:
            return 1 if self.major > other.major else -1
        if self.minor != other.minor:
            return 1 if self.minor > other.minor else -1
        if self.patch != other.patch:
            return 1 if self.patch > other.patch else -1
        return 0
    
    def is_compatible(self, other: "Version") -> bool:
        """
        Verificar compatibilidad (mismo major).
        
        Args:
            other: Otra versión
            
        Returns:
            True si son compatibles
        """
        return self.major == other.major


@dataclass
class VersionInfo:
    """Información completa de versión."""
    version: Version
    release_date: Optional[str] = None
    changelog: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None
    features: Optional[List[str]] = None
    bugfixes: Optional[List[str]] = None
    breaking_changes: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "version": str(self.version),
            "version_details": self.version.to_dict(),
            "release_date": self.release_date,
            "changelog": self.changelog,
            "dependencies": self.dependencies,
            "features": self.features,
            "bugfixes": self.bugfixes,
            "breaking_changes": self.breaking_changes
        }


class VersionManager:
    """
    Gestor de versiones.
    
    Gestiona versiones del sistema y actualizaciones.
    """
    
    # Versión actual del sistema
    CURRENT_VERSION = Version(1, 0, 0, build="2024.12")
    
    def __init__(self):
        """Inicializar gestor de versiones."""
        self.version_history: List[VersionInfo] = []
        self.current_version_info = VersionInfo(
            version=self.CURRENT_VERSION,
            release_date=datetime.now().isoformat(),
            features=[
                "Trajectory optimization with RL",
                "Multi-algorithm support (PPO, DQN, A*, RRT)",
                "Real-time feedback system",
                "Chat-based robot control",
                "ROS integration",
                "Multi-robot support",
                "Advanced caching",
                "Event system",
                "Health checks",
                "Resource monitoring",
                "Quality assurance"
            ]
        )
    
    def get_current_version(self) -> Version:
        """Obtener versión actual."""
        return self.CURRENT_VERSION
    
    def get_version_info(self) -> VersionInfo:
        """Obtener información de versión actual."""
        return self.current_version_info
    
    def check_update(self, latest_version: Version) -> Dict[str, Any]:
        """
        Verificar si hay actualizaciones.
        
        Args:
            latest_version: Última versión disponible
            
        Returns:
            Información de actualización
        """
        comparison = self.CURRENT_VERSION.compare(latest_version)
        
        return {
            "current_version": str(self.CURRENT_VERSION),
            "latest_version": str(latest_version),
            "update_available": comparison < 0,
            "is_newer": comparison > 0,
            "is_compatible": self.CURRENT_VERSION.is_compatible(latest_version),
            "update_type": self._get_update_type(self.CURRENT_VERSION, latest_version)
        }
    
    def _get_update_type(self, current: Version, latest: Version) -> str:
        """Obtener tipo de actualización."""
        if latest.major > current.major:
            return "major"
        elif latest.minor > current.minor:
            return "minor"
        elif latest.patch > current.patch:
            return "patch"
        else:
            return "none"
    
    def get_migration_path(self, from_version: Version, to_version: Version) -> List[str]:
        """
        Obtener ruta de migración entre versiones.
        
        Args:
            from_version: Versión origen
            to_version: Versión destino
            
        Returns:
            Lista de pasos de migración
        """
        steps = []
        
        if to_version.major > from_version.major:
            steps.append(f"Major version update: {from_version.major} -> {to_version.major}")
            steps.append("Review breaking changes")
            steps.append("Update configuration")
            steps.append("Test thoroughly")
        
        if to_version.minor > from_version.minor:
            steps.append(f"Minor version update: {from_version.minor} -> {to_version.minor}")
            steps.append("Review new features")
            steps.append("Update dependencies if needed")
        
        if to_version.patch > from_version.patch:
            steps.append(f"Patch version update: {from_version.patch} -> {to_version.patch}")
            steps.append("Apply bugfixes")
        
        return steps
    
    def export_version_info(self, filepath: str) -> None:
        """
        Exportar información de versión.
        
        Args:
            filepath: Ruta del archivo
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_version_info.to_dict(), f, indent=2)
            logger.info(f"Version info exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting version info: {e}")


# Instancia global
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Obtener instancia global del gestor de versiones."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager


def get_current_version() -> Version:
    """Obtener versión actual del sistema."""
    return get_version_manager().get_current_version()






