"""
Sistema de versionado de modelos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json


class ModelStatus(str, Enum):
    """Estado del modelo"""
    DRAFT = "draft"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """Versión de modelo"""
    version: str
    model_path: str
    status: ModelStatus
    description: Optional[str] = None
    metadata: Optional[Dict] = None
    created_at: str = None
    performance_metrics: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "status": self.status.value,
            "description": self.description,
            "metadata": self.metadata or {},
            "created_at": self.created_at,
            "performance_metrics": self.performance_metrics or {}
        }


class ModelVersioning:
    """Sistema de versionado de modelos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.models: Dict[str, List[ModelVersion]] = {}  # model_name -> versions
    
    def register_model_version(self, model_name: str, version: str,
                              model_path: str, status: ModelStatus = ModelStatus.DRAFT,
                              description: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> ModelVersion:
        """
        Registra una versión de modelo
        
        Args:
            model_name: Nombre del modelo
            version: Versión
            model_path: Path al modelo
            status: Estado
            description: Descripción
            metadata: Metadatos
            
        Returns:
            Versión del modelo
        """
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            status=status,
            description=description,
            metadata=metadata
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        self.models[model_name].append(model_version)
        return model_version
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Obtiene versiones de un modelo"""
        return self.models.get(model_name, [])
    
    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Obtiene versión en producción"""
        versions = self.get_model_versions(model_name)
        
        for version in versions:
            if version.status == ModelStatus.PRODUCTION:
                return version
        
        return None
    
    def set_model_status(self, model_name: str, version: str, status: ModelStatus):
        """
        Cambia estado de un modelo
        
        Args:
            model_name: Nombre del modelo
            version: Versión
            status: Nuevo estado
        """
        versions = self.get_model_versions(model_name)
        
        for v in versions:
            if v.version == version:
                v.status = status
                break
    
    def update_performance_metrics(self, model_name: str, version: str,
                                  metrics: Dict):
        """
        Actualiza métricas de rendimiento
        
        Args:
            model_name: Nombre del modelo
            version: Versión
            metrics: Métricas
        """
        versions = self.get_model_versions(model_name)
        
        for v in versions:
            if v.version == version:
                v.performance_metrics = metrics
                break
    
    def get_model_info(self, model_name: str) -> Dict:
        """Obtiene información de un modelo"""
        versions = self.get_model_versions(model_name)
        
        return {
            "model_name": model_name,
            "total_versions": len(versions),
            "versions": [v.to_dict() for v in versions],
            "production_version": self.get_production_version(model_name).to_dict() if self.get_production_version(model_name) else None
        }






