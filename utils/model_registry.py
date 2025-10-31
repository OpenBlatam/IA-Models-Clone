from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import CACHE_TTL, VALIDATION_TIMEOUT
from .model_field import ModelField, FieldConfig
from .model_schema import ModelSchema, SchemaConfig
from .model_factory import ModelFactory
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Registry - Onyx Integration
Registry for managing models and schemas.
"""

T = TypeVar('T')

class ModelRegistry:
    """Registry for managing models and schemas."""
    
    def __init__(self) -> Any:
        """Initialize registry."""
        self._factory = ModelFactory()
        self._instances: Dict[str, Dict[str, 'OnyxBaseModel']] = {}
    
    def register_schema(self, name: str, schema: ModelSchema) -> None:
        """Register a schema."""
        self._factory.register_schema(name, schema)
    
    def register_model(self, name: str, model_class: 'OnyxBaseModel') -> None:
        self._factory.register_model(name, model_class)
        self._instances[name] = {}
    
    def get_schema(self, name: str) -> Optional[ModelSchema]:
        """Get a schema."""
        return self._factory.get_schema(name)
    
    def get_model_class(self, name: str) -> Optional[Type['OnyxBaseModel']]:
        return self._factory.get_model_class(name)
    
    def create_model(self, name: str, data: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> Optional['OnyxBaseModel']:
        model = self._factory.create_model(name, data, id)
        
        if model and model.id:
            self._instances[name][model.id] = model
        
        return model
    
    def get_model(self, name: str, id: str) -> Optional['OnyxBaseModel']:
        return self._instances.get(name, {}).get(id)
    
    def get_models(self, name: str) -> Dict[str, 'OnyxBaseModel']:
        return self._instances.get(name, {}).copy()
    
    def update_model(self, name: str, id: str, data: Dict[str, Any]) -> Optional['OnyxBaseModel']:
        model = self.get_model(name, id)
        
        if model:
            model.set_data(data)
        
        return model
    
    def delete_model(self, name: str, id: str) -> None:
        """Delete a model."""
        if name in self._instances:
            self._instances[name].pop(id, None)
    
    def clear_models(self, name: str) -> None:
        """Clear all models of a type."""
        if name in self._instances:
            self._instances[name].clear()
    
    def clear_all_models(self) -> None:
        """Clear all models."""
        self._instances.clear()
    
    def get_model_count(self, name: str) -> int:
        """Get count of models of a type."""
        return len(self._instances.get(name, {}))
    
    def get_total_model_count(self) -> int:
        """Get total count of all models."""
        return sum(len(models) for models in self._instances.values())
    
    def get_model_ids(self, name: str) -> List[str]:
        """Get IDs of all models of a type."""
        return list(self._instances.get(name, {}).keys())
    
    def get_all_model_ids(self) -> Dict[str, List[str]]:
        """Get IDs of all models."""
        return {
            name: list(models.keys())
            for name, models in self._instances.items()
        }
    
    def get_model_data(self, name: str, id: str) -> Optional[Dict[str, Any]]:
        """Get data of a model."""
        model = self.get_model(name, id)
        return model.get_data() if model else None
    
    def get_models_data(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Get data of all models of a type."""
        return {
            id: model.get_data()
            for id, model in self._instances.get(name, {}).items()
        }
    
    def get_all_models_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get data of all models."""
        return {
            name: {
                id: model.get_data()
                for id, model in models.items()
            }
            for name, models in self._instances.items()
        } 