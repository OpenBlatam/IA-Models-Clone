"""
Model Manager
=============

ML model management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class Model:
    """ML model definition."""
    id: str
    name: str
    version: str
    type: str  # classification, regression, etc.
    status: ModelStatus
    path: str
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    accuracy: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class ModelManager:
    """ML model manager."""
    
    def __init__(self):
        self._models: Dict[str, Model] = {}
        self._deployed_models: Dict[str, str] = {}  # model_id -> version
    
    def register_model(
        self,
        model_id: str,
        name: str,
        version: str,
        model_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Model:
        """Register model."""
        model = Model(
            id=model_id,
            name=name,
            version=version,
            type=model_type,
            status=ModelStatus.READY,
            path=path,
            metadata=metadata or {}
        )
        
        self._models[model_id] = model
        logger.info(f"Registered model: {model_id} v{version}")
        return model
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def deploy_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Deploy model."""
        if model_id not in self._models:
            return False
        
        model = self._models[model_id]
        
        if version and model.version != version:
            return False
        
        model.status = ModelStatus.DEPLOYED
        self._deployed_models[model_id] = model.version
        
        logger.info(f"Deployed model: {model_id} v{model.version}")
        return True
    
    def undeploy_model(self, model_id: str) -> bool:
        """Undeploy model."""
        if model_id not in self._models:
            return False
        
        model = self._models[model_id]
        model.status = ModelStatus.READY
        self._deployed_models.pop(model_id, None)
        
        logger.info(f"Undeployed model: {model_id}")
        return True
    
    def update_model_accuracy(self, model_id: str, accuracy: float):
        """Update model accuracy."""
        if model_id in self._models:
            self._models[model_id].accuracy = accuracy
            logger.info(f"Updated accuracy for {model_id}: {accuracy:.4f}")
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[Model]:
        """List models."""
        models = list(self._models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    def get_deployed_models(self) -> List[Model]:
        """Get deployed models."""
        return [
            self._models[model_id]
            for model_id in self._deployed_models.keys()
            if model_id in self._models
        ]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "total_models": len(self._models),
            "deployed_models": len(self._deployed_models),
            "by_status": {
                status.value: sum(1 for m in self._models.values() if m.status == status)
                for status in ModelStatus
            },
            "by_type": {
                model_type: sum(1 for m in self._models.values() if m.type == model_type)
                for model_type in set(m.type for m in self._models.values())
            }
        }















