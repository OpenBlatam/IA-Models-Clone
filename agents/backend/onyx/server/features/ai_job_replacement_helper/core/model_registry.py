"""
Model Registry Service - Registro de modelos
============================================

Sistema centralizado para registrar y gestionar modelos.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Estados de modelo"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadatos de modelo"""
    model_id: str
    name: str
    version: str
    status: ModelStatus
    model_type: str
    created_at: datetime
    created_by: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class ModelRegistryService:
    """Servicio de registro de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.models: Dict[str, ModelMetadata] = {}
        logger.info("ModelRegistryService initialized")
    
    def register_model(
        self,
        model_id: str,
        name: str,
        version: str,
        model_type: str,
        status: ModelStatus = ModelStatus.DEVELOPMENT,
        description: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        created_by: str = ""
    ) -> ModelMetadata:
        """Registrar nuevo modelo"""
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            status=status,
            model_type=model_type,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            metrics=metrics or {},
            config=config or {},
        )
        
        self.models[model_id] = metadata
        
        logger.info(f"Model {model_id} registered")
        return metadata
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Obtener modelo por ID"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        model_type: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[ModelMetadata]:
        """Listar modelos con filtros"""
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tag:
            models = [m for m in models if tag in m.tags]
        
        return models
    
    def update_model_status(
        self,
        model_id: str,
        new_status: ModelStatus
    ) -> bool:
        """Actualizar estado del modelo"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        old_status = model.status
        model.status = new_status
        
        logger.info(f"Model {model_id} status changed: {old_status} -> {new_status}")
        return True
    
    def update_model_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Actualizar métricas del modelo"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        model.metrics.update(metrics)
        logger.info(f"Model {model_id} metrics updated")
        return True
    
    def promote_to_production(
        self,
        model_id: str
    ) -> bool:
        """Promover modelo a producción"""
        return self.update_model_status(model_id, ModelStatus.PRODUCTION)
    
    def archive_model(self, model_id: str) -> bool:
        """Archivar modelo"""
        return self.update_model_status(model_id, ModelStatus.ARCHIVED)




