"""
Model Registry - Registro y gestión de modelos
===============================================
"""

import logging
import torch
import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Estados de modelo"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata de modelo"""
    model_id: str
    name: str
    version: str
    model_type: str
    framework: str = "pytorch"
    status: ModelStatus = ModelStatus.READY
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "framework": self.framework,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_path": self.file_path,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "tags": self.tags,
            "description": self.description
        }


class ModelRegistry:
    """Registro de modelos"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.models: Dict[str, ModelMetadata] = {}
        os.makedirs(registry_path, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        """Carga el registro desde disco"""
        registry_file = os.path.join(self.registry_path, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for model_data in data.get("models", []):
                        metadata = ModelMetadata(
                            model_id=model_data["model_id"],
                            name=model_data["name"],
                            version=model_data["version"],
                            model_type=model_data["model_type"],
                            framework=model_data.get("framework", "pytorch"),
                            status=ModelStatus(model_data["status"]),
                            created_at=datetime.fromisoformat(model_data["created_at"]),
                            updated_at=datetime.fromisoformat(model_data["updated_at"]),
                            file_path=model_data.get("file_path"),
                            file_size=model_data.get("file_size"),
                            checksum=model_data.get("checksum"),
                            hyperparameters=model_data.get("hyperparameters", {}),
                            metrics=model_data.get("metrics", {}),
                            tags=model_data.get("tags", []),
                            description=model_data.get("description", "")
                        )
                        self.models[metadata.model_id] = metadata
            except Exception as e:
                logger.error(f"Error cargando registro: {e}")
    
    def _save_registry(self):
        """Guarda el registro en disco"""
        registry_file = os.path.join(self.registry_path, "registry.json")
        try:
            data = {
                "models": [model.to_dict() for model in self.models.values()]
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando registro: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcula checksum de un archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_type: str,
        model_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> ModelMetadata:
        """Registra un modelo"""
        import uuid
        
        model_id = str(uuid.uuid4())
        
        # Calcular checksum y tamaño
        file_size = os.path.getsize(model_path) if os.path.exists(model_path) else None
        checksum = self._calculate_checksum(model_path) if os.path.exists(model_path) else None
        
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            file_path=model_path,
            file_size=file_size,
            checksum=checksum,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            tags=tags or [],
            description=description
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        logger.info(f"Modelo {name} v{version} registrado: {model_id}")
        return metadata
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Obtiene metadata de un modelo"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """Lista modelos con filtros"""
        models = list(self.models.values())
        
        if name:
            models = [m for m in models if name.lower() in m.name.lower()]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return models
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """Actualiza el estado de un modelo"""
        if model_id in self.models:
            self.models[model_id].status = status
            self.models[model_id].updated_at = datetime.now()
            self._save_registry()
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Actualiza métricas de un modelo"""
        if model_id in self.models:
            self.models[model_id].metrics.update(metrics)
            self.models[model_id].updated_at = datetime.now()
            self._save_registry()
    
    def load_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Carga un modelo desde el registro"""
        metadata = self.get_model(model_id)
        if not metadata or not metadata.file_path:
            return None
        
        if not os.path.exists(metadata.file_path):
            logger.error(f"Archivo de modelo no encontrado: {metadata.file_path}")
            return None
        
        try:
            model = torch.load(metadata.file_path, map_location="cpu")
            logger.info(f"Modelo {model_id} cargado")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")
            return None




