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
from .model_registry import ModelRegistry
from .model_manager import ModelManager
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Service - Onyx Integration
Service for handling model operations with caching and validation.
"""

T = TypeVar('T')

class ModelService:
    """Service for handling model operations with caching and validation."""
    
    def __init__(self) -> Any:
        """Initialize service."""
        self._manager = ModelManager()
        self._validation_cache: Dict[str, Dict[str, bool]] = {}
        self._validation_timestamps: Dict[str, Dict[str, float]] = {}
    
    def register_schema(self, name: str, schema: ModelSchema) -> None:
        """Register a schema."""
        self._manager.register_schema(name, schema)
        self._validation_cache[name] = {}
        self._validation_timestamps[name] = {}
    
    def register_model(self, name: str, model_class: Type['OnyxBaseModel']) -> None:
        """Register a model class."""
        self._manager.register_model(name, model_class)
    
    def create_model(self, name: str, data: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> Optional['OnyxBaseModel']:
        """Create a model."""
        # Validate data
        schema = self._manager.get_schema(name)
        if schema:
            errors = schema.validate(data or {})
            if errors:
                raise ValueError(f"Invalid data: {', '.join(errors)}")
        
        # Create model
        model = self._manager.create_model(name, data, id)
        
        if model and model.id:
            self._validation_cache[name][model.id] = True
            self._validation_timestamps[name][model.id] = datetime.utcnow().timestamp()
        
        return model
    
    def get_model(self, name: str, id: str) -> Optional['OnyxBaseModel']:
        """Get a model by ID."""
        # Check validation cache
        if name in self._validation_cache and id in self._validation_cache[name]:
            timestamp = self._validation_timestamps[name][id]
            if datetime.utcnow().timestamp() - timestamp <= VALIDATION_TIMEOUT:
                return self._manager.get_model(name, id)
        
        # Get model
        model = self._manager.get_model(name, id)
        
        if model:
            # Validate model
            errors = model.validate()
            if not errors:
                self._validation_cache[name][id] = True
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
            else:
                self._validation_cache[name][id] = False
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
                raise ValueError(f"Invalid model: {', '.join(errors)}")
        
        return model
    
    def get_models(self, name: str) -> Dict[str, 'OnyxBaseModel']:
        """Get all models of a type."""
        models = self._manager.get_models(name)
        
        # Validate models
        for id, model in models.items():
            errors = model.validate()
            if not errors:
                self._validation_cache[name][id] = True
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
            else:
                self._validation_cache[name][id] = False
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
                raise ValueError(f"Invalid model {id}: {', '.join(errors)}")
        
        return models
    
    def update_model(self, name: str, id: str, data: Dict[str, Any]) -> Optional['OnyxBaseModel']:
        """Update a model."""
        # Validate data
        schema = self._manager.get_schema(name)
        if schema:
            errors = schema.validate(data)
            if errors:
                raise ValueError(f"Invalid data: {', '.join(errors)}")
        
        # Update model
        model = self._manager.update_model(name, id, data)
        
        if model:
            # Validate model
            errors = model.validate()
            if not errors:
                self._validation_cache[name][id] = True
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
            else:
                self._validation_cache[name][id] = False
                self._validation_timestamps[name][id] = datetime.utcnow().timestamp()
                raise ValueError(f"Invalid model: {', '.join(errors)}")
        
        return model
    
    def delete_model(self, name: str, id: str) -> None:
        """Delete a model."""
        self._manager.delete_model(name, id)
        
        if name in self._validation_cache:
            self._validation_cache[name].pop(id, None)
            self._validation_timestamps[name].pop(id, None)
    
    def clear_models(self, name: str) -> None:
        """Clear all models of a type."""
        self._manager.clear_models(name)
        
        if name in self._validation_cache:
            self._validation_cache[name].clear()
            self._validation_timestamps[name].clear()
    
    def clear_all_models(self) -> None:
        """Clear all models."""
        self._manager.clear_all_models()
        self._validation_cache.clear()
        self._validation_timestamps.clear()
    
    def get_model_count(self, name: str) -> int:
        """Get count of models of a type."""
        return self._manager.get_model_count(name)
    
    def get_total_model_count(self) -> int:
        """Get total count of all models."""
        return self._manager.get_total_model_count()
    
    def get_model_ids(self, name: str) -> List[str]:
        """Get IDs of all models of a type."""
        return self._manager.get_model_ids(name)
    
    def get_all_model_ids(self) -> Dict[str, List[str]]:
        """Get IDs of all models."""
        return self._manager.get_all_model_ids()
    
    def get_model_data(self, name: str, id: str) -> Optional[Dict[str, Any]]:
        """Get data of a model."""
        return self._manager.get_model_data(name, id)
    
    def get_models_data(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Get data of all models of a type."""
        return self._manager.get_models_data(name)
    
    def get_all_models_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get data of all models."""
        return self._manager.get_all_models_data()
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cache."""
        self._manager.clear_cache(name)
        
        if name:
            if name in self._validation_cache:
                self._validation_cache[name].clear()
                self._validation_timestamps[name].clear()
        else:
            self._validation_cache.clear()
            self._validation_timestamps.clear()
    
    def get_cache_size(self, name: Optional[str] = None) -> int:
        """Get cache size."""
        return self._manager.get_cache_size(name)
    
    def get_validation_cache_size(self, name: Optional[str] = None) -> int:
        """Get validation cache size."""
        if name:
            return len(self._validation_cache.get(name, {}))
        return sum(len(cache) for cache in self._validation_cache.values())
    
    def get_cache_hits(self, name: Optional[str] = None) -> int:
        """Get cache hits."""
        return self._manager.get_cache_hits(name)
    
    def get_cache_misses(self, name: Optional[str] = None) -> int:
        """Get cache misses."""
        return self._manager.get_cache_misses(name)
    
    def get_validation_hits(self, name: Optional[str] = None) -> int:
        """Get validation hits."""
        # TODO: Implement validation hit tracking
        return 0
    
    def get_validation_misses(self, name: Optional[str] = None) -> int:
        """Get validation misses."""
        # TODO: Implement validation miss tracking
        return 0 