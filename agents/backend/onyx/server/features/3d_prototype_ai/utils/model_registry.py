"""
Model Registry - Registro y versionado de modelos
==================================================
Gestión de versiones, metadata, y búsqueda de modelos
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Estados de modelo"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata de modelo"""
    model_id: str
    version: str
    name: str
    description: str
    architecture: str
    status: ModelStatus
    created_at: str
    updated_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    dataset_info: Dict[str, Any]
    tags: List[str]
    author: str
    model_path: str
    checkpoint_path: Optional[str] = None


class ModelRegistry:
    """Registro de modelos"""
    
    def __init__(self, registry_dir: str = "./storage/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Carga registro desde disco"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for model_id, meta_dict in data.items():
                        meta_dict["status"] = ModelStatus(meta_dict["status"])
                        self.models[model_id] = ModelMetadata(**meta_dict)
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Guarda registro a disco"""
        data = {
            model_id: asdict(metadata)
            for model_id, metadata in self.models.items()
        }
        
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model_id: str,
        name: str,
        architecture: str,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        author: str = "system",
        model_path: str = "",
        checkpoint_path: Optional[str] = None
    ) -> ModelMetadata:
        """Registra un nuevo modelo"""
        # Generar versión
        version = self._generate_version(model_id)
        
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            architecture=architecture,
            status=ModelStatus.TRAINED,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            dataset_info=dataset_info or {},
            tags=tags or [],
            author=author,
            model_path=model_path,
            checkpoint_path=checkpoint_path
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {model_id} v{version}")
        return metadata
    
    def _generate_version(self, model_id: str) -> str:
        """Genera versión para modelo"""
        # Contar versiones existentes
        existing_versions = [
            m.version for m in self.models.values()
            if m.model_id == model_id
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Incrementar versión menor
        latest = existing_versions[-1]
        parts = latest.split(".")
        parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """Actualiza estado de modelo"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.models[model_id].status = status
        self.models[model_id].updated_at = datetime.now().isoformat()
        self._save_registry()
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Actualiza métricas de modelo"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.models[model_id].metrics.update(metrics)
        self.models[model_id].updated_at = datetime.now().isoformat()
        self._save_registry()
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Obtiene metadata de modelo"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        architecture: Optional[str] = None
    ) -> List[ModelMetadata]:
        """Lista modelos con filtros"""
        filtered = list(self.models.values())
        
        if status:
            filtered = [m for m in filtered if m.status == status]
        
        if tags:
            filtered = [m for m in filtered if any(tag in m.tags for tag in tags)]
        
        if architecture:
            filtered = [m for m in filtered if m.architecture == architecture]
        
        return sorted(filtered, key=lambda x: x.updated_at, reverse=True)
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """Busca modelos por query"""
        query_lower = query.lower()
        results = []
        
        for model in self.models.values():
            if (query_lower in model.name.lower() or
                query_lower in model.description.lower() or
                any(query_lower in tag.lower() for tag in model.tags)):
                results.append(model)
        
        return results
    
    def delete_model(self, model_id: str):
        """Elimina modelo del registro"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        del self.models[model_id]
        self._save_registry()
        logger.info(f"Deleted model: {model_id}")




