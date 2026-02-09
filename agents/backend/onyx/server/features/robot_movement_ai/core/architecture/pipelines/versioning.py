"""
Pipeline Versioning
===================

Sistema de versionado para pipelines.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .pipeline import Pipeline
from .stages import PipelineStage

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Tipo de versión."""
    MAJOR = "major"  # Cambios incompatibles
    MINOR = "minor"  # Nuevas funcionalidades compatibles
    PATCH = "patch"  # Correcciones de bugs


@dataclass
class PipelineVersion:
    """
    Versión de pipeline.
    """
    version: str
    version_type: VersionType
    description: str
    created_at: str
    stages: List[str] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Representación string."""
        return f"v{self.version} ({self.version_type.value})"


class VersionManager:
    """
    Gestor de versiones para pipelines.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.versions: Dict[str, List[PipelineVersion]] = {}
    
    def create_version(
        self,
        pipeline_name: str,
        version_type: VersionType = VersionType.PATCH,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineVersion:
        """
        Crear nueva versión.
        
        Args:
            pipeline_name: Nombre del pipeline
            version_type: Tipo de versión
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            Nueva versión
        """
        # Obtener versión anterior
        current_version = self.get_latest_version(pipeline_name)
        
        # Calcular nueva versión
        if current_version:
            major, minor, patch = map(int, current_version.version.split('.'))
            
            if version_type == VersionType.MAJOR:
                major += 1
                minor = 0
                patch = 0
            elif version_type == VersionType.MINOR:
                minor += 1
                patch = 0
            else:  # PATCH
                patch += 1
            
            new_version = f"{major}.{minor}.{patch}"
        else:
            new_version = "1.0.0"
        
        version = PipelineVersion(
            version=new_version,
            version_type=version_type,
            description=description,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        if pipeline_name not in self.versions:
            self.versions[pipeline_name] = []
        
        self.versions[pipeline_name].append(version)
        
        logger.info(f"Versión creada: {pipeline_name} {version}")
        return version
    
    def get_latest_version(self, pipeline_name: str) -> Optional[PipelineVersion]:
        """
        Obtener última versión.
        
        Args:
            pipeline_name: Nombre del pipeline
            
        Returns:
            Última versión o None
        """
        if pipeline_name not in self.versions or not self.versions[pipeline_name]:
            return None
        
        return self.versions[pipeline_name][-1]
    
    def get_version(self, pipeline_name: str, version: str) -> Optional[PipelineVersion]:
        """
        Obtener versión específica.
        
        Args:
            pipeline_name: Nombre del pipeline
            version: Versión
            
        Returns:
            Versión o None
        """
        if pipeline_name not in self.versions:
            return None
        
        for v in self.versions[pipeline_name]:
            if v.version == version:
                return v
        
        return None
    
    def list_versions(self, pipeline_name: str) -> List[PipelineVersion]:
        """
        Listar todas las versiones.
        
        Args:
            pipeline_name: Nombre del pipeline
            
        Returns:
            Lista de versiones
        """
        return self.versions.get(pipeline_name, [])


class VersionedPipeline(Pipeline):
    """
    Pipeline con soporte para versionado.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        version_manager: Optional[VersionManager] = None,
        **kwargs
    ):
        """
        Inicializar pipeline versionado.
        
        Args:
            name: Nombre del pipeline
            version_manager: Gestor de versiones
            **kwargs: Argumentos adicionales
        """
        super().__init__(name, **kwargs)
        self.version_manager = version_manager or VersionManager()
        self._current_version: Optional[PipelineVersion] = None
    
    def snapshot(self) -> PipelineVersion:
        """
        Crear snapshot del pipeline actual.
        
        Returns:
            Versión creada
        """
        version = self.version_manager.create_version(
            pipeline_name=self.name,
            version_type=VersionType.PATCH,
            description="Snapshot automático",
            metadata={
                'stage_count': len(self.stages),
                'middleware_count': len(self.middleware)
            }
        )
        
        # Guardar información de etapas y middleware
        version.stages = [stage.get_name() for stage in self.stages]
        version.middleware = [mw.__class__.__name__ for mw in self.middleware]
        
        self._current_version = version
        return version
    
    def get_version(self) -> Optional[str]:
        """
        Obtener versión actual.
        
        Returns:
            Versión o None
        """
        if self._current_version:
            return self._current_version.version
        
        latest = self.version_manager.get_latest_version(self.name)
        return latest.version if latest else None
    
    def get_version_info(self) -> Optional[PipelineVersion]:
        """
        Obtener información de versión.
        
        Returns:
            Información de versión o None
        """
        if self._current_version:
            return self._current_version
        
        return self.version_manager.get_latest_version(self.name)

