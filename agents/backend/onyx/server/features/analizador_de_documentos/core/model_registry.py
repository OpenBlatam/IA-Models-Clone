"""
Sistema de Model Registry
===========================

Sistema para registro de modelos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Etapa del modelo"""
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Versión de modelo"""
    version_id: str
    model_id: str
    version: str
    stage: ModelStage
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    created_at: str
    created_by: str


@dataclass
class RegisteredModel:
    """Modelo registrado"""
    model_id: str
    name: str
    description: str
    versions: List[str]
    latest_version: str
    created_at: str


class ModelRegistry:
    """
    Sistema de Model Registry
    
    Proporciona:
    - Registro de modelos
    - Versionado de modelos
    - Gestión de etapas (Staging, Production, Archived)
    - Metadata de modelos
    - Búsqueda de modelos
    - Comparación de versiones
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.models: Dict[str, RegisteredModel] = {}
        self.versions: Dict[str, ModelVersion] = {}
        logger.info("ModelRegistry inicializado")
    
    def register_model(
        self,
        name: str,
        description: str = ""
    ) -> RegisteredModel:
        """
        Registrar modelo
        
        Args:
            name: Nombre del modelo
            description: Descripción
        
        Returns:
            Modelo registrado
        """
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model = RegisteredModel(
            model_id=model_id,
            name=name,
            description=description,
            versions=[],
            latest_version="1.0.0",
            created_at=datetime.now().isoformat()
        )
        
        self.models[model_id] = model
        
        logger.info(f"Modelo registrado: {model_id}")
        
        return model
    
    def register_version(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float],
        stage: ModelStage = ModelStage.NONE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Registrar versión de modelo
        
        Args:
            model_id: ID del modelo
            version: Versión
            metrics: Métricas
            stage: Etapa
            metadata: Metadata
        
        Returns:
            Versión registrada
        """
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        version_id = f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version=version,
            stage=stage,
            metrics=metrics,
            metadata=metadata or {},
            created_at=datetime.now().isoformat(),
            created_by="system"
        )
        
        self.versions[version_id] = model_version
        
        model = self.models[model_id]
        model.versions.append(version)
        model.latest_version = version
        
        logger.info(f"Versión registrada: {version_id} - {version}")
        
        return model_version
    
    def transition_stage(
        self,
        model_id: str,
        version: str,
        new_stage: ModelStage
    ) -> ModelVersion:
        """
        Transicionar etapa de modelo
        
        Args:
            model_id: ID del modelo
            version: Versión
            new_stage: Nueva etapa
        
        Returns:
            Versión actualizada
        """
        # Buscar versión
        version_obj = None
        for v in self.versions.values():
            if v.model_id == model_id and v.version == version:
                version_obj = v
                break
        
        if version_obj is None:
            raise ValueError(f"Versión no encontrada: {version}")
        
        version_obj.stage = new_stage
        
        logger.info(f"Etapa transicionada: {model_id} - {version} -> {new_stage.value}")
        
        return version_obj


# Instancia global
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Obtener instancia global del sistema"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


