"""
AI Model Entity

This module defines the AI model entity that represents a machine learning model
in the HeyGen AI system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base_entity import BaseEntity


class ModelType(str, Enum):
    """Enumeration of supported model types."""
    TRANSFORMER = "transformer"
    QUANTUM = "quantum"
    NEUROMORPHIC = "neuromorphic"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Enumeration of model statuses."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AIModel(BaseEntity):
    """
    AI Model entity representing a machine learning model.
    
    This entity contains all the information about a model including:
    - Model metadata (name, type, version)
    - Model configuration
    - Training status and metrics
    - Deployment information
    """
    
    def __init__(
        self,
        name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        description: Optional[str] = None,
        model_id: Optional[UUID] = None
    ):
        """
        Initialize the AI model entity.
        
        Args:
            name: Name of the model
            model_type: Type of the model
            version: Version of the model
            description: Optional description of the model
            model_id: Optional UUID for the model
        """
        super().__init__(model_id)
        
        self._name = name
        self._model_type = model_type
        self._version = version
        self._description = description
        self._status = ModelStatus.TRAINING
        self._config: Dict[str, Any] = {}
        self._metrics: Dict[str, float] = {}
        self._tags: List[str] = []
        self._path: Optional[str] = None
        self._size_bytes: Optional[int] = None
        self._created_by: Optional[str] = None
        self._last_trained_at: Optional[datetime] = None
        self._last_deployed_at: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        """Get the model name."""
        return self._name
    
    @property
    def model_type(self) -> ModelType:
        """Get the model type."""
        return self._model_type
    
    @property
    def version(self) -> str:
        """Get the model version."""
        return self._version
    
    @property
    def description(self) -> Optional[str]:
        """Get the model description."""
        return self._description
    
    @property
    def status(self) -> ModelStatus:
        """Get the model status."""
        return self._status
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self._config.copy()
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Get the model metrics."""
        return self._metrics.copy()
    
    @property
    def tags(self) -> List[str]:
        """Get the model tags."""
        return self._tags.copy()
    
    @property
    def path(self) -> Optional[str]:
        """Get the model file path."""
        return self._path
    
    @property
    def size_bytes(self) -> Optional[int]:
        """Get the model size in bytes."""
        return self._size_bytes
    
    @property
    def created_by(self) -> Optional[str]:
        """Get the creator of the model."""
        return self._created_by
    
    @property
    def last_trained_at(self) -> Optional[datetime]:
        """Get the last training timestamp."""
        return self._last_trained_at
    
    @property
    def last_deployed_at(self) -> Optional[datetime]:
        """Get the last deployment timestamp."""
        return self._last_deployed_at
    
    def set_name(self, name: str) -> None:
        """Set the model name."""
        self._name = name
        self.touch()
    
    def set_description(self, description: str) -> None:
        """Set the model description."""
        self._description = description
        self.touch()
    
    def set_status(self, status: ModelStatus) -> None:
        """Set the model status."""
        self._status = status
        self.touch()
        
        if status == ModelStatus.TRAINED:
            self._last_trained_at = datetime.utcnow()
        elif status == ModelStatus.DEPLOYED:
            self._last_deployed_at = datetime.utcnow()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the model configuration."""
        self._config.update(config)
        self.touch()
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update the model metrics."""
        self._metrics.update(metrics)
        self.touch()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self._tags:
            self._tags.append(tag)
            self.touch()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self._tags:
            self._tags.remove(tag)
            self.touch()
    
    def set_path(self, path: str) -> None:
        """Set the model file path."""
        self._path = path
        self.touch()
    
    def set_size_bytes(self, size_bytes: int) -> None:
        """Set the model size in bytes."""
        self._size_bytes = size_bytes
        self.touch()
    
    def set_created_by(self, created_by: str) -> None:
        """Set the creator of the model."""
        self._created_by = created_by
        self.touch()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity to a dictionary representation."""
        return {
            'id': str(self._id),
            'name': self._name,
            'model_type': self._model_type.value,
            'version': self._version,
            'description': self._description,
            'status': self._status.value,
            'config': self._config,
            'metrics': self._metrics,
            'tags': self._tags,
            'path': self._path,
            'size_bytes': self._size_bytes,
            'created_by': self._created_by,
            'created_at': self._created_at.isoformat(),
            'updated_at': self._updated_at.isoformat(),
            'version': self._version,
            'last_trained_at': self._last_trained_at.isoformat() if self._last_trained_at else None,
            'last_deployed_at': self._last_deployed_at.isoformat() if self._last_deployed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIModel':
        """Create an entity from a dictionary representation."""
        model = cls(
            name=data['name'],
            model_type=ModelType(data['model_type']),
            version=data['version'],
            description=data.get('description'),
            model_id=UUID(data['id'])
        )
        
        # Set timestamps
        model._created_at = datetime.fromisoformat(data['created_at'])
        model._updated_at = datetime.fromisoformat(data['updated_at'])
        model._version = data['version']
        
        # Set optional fields
        if 'status' in data:
            model._status = ModelStatus(data['status'])
        if 'config' in data:
            model._config = data['config']
        if 'metrics' in data:
            model._metrics = data['metrics']
        if 'tags' in data:
            model._tags = data['tags']
        if 'path' in data:
            model._path = data['path']
        if 'size_bytes' in data:
            model._size_bytes = data['size_bytes']
        if 'created_by' in data:
            model._created_by = data['created_by']
        if 'last_trained_at' in data and data['last_trained_at']:
            model._last_trained_at = datetime.fromisoformat(data['last_trained_at'])
        if 'last_deployed_at' in data and data['last_deployed_at']:
            model._last_deployed_at = datetime.fromisoformat(data['last_deployed_at'])
        
        return model