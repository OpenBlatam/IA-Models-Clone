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
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Manager - Onyx Integration
Manager for handling model operations.
"""

T = TypeVar('T')

class ModelManager:
    """Manager for handling model operations."""
    
    def __init__(self) -> Any:
        """Initialize manager."""
        self._registry = ModelRegistry()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, Dict[str, float]] = {}
    
    def register_schema(self, name: str, schema: ModelSchema) -> None:
        """Register a schema."""
        self._registry.register_schema(name, schema)
    
    def register_model(self, name: str, model_class: Type['OnyxBaseModel']) -> None:
        """Register a model class."""
        # Import here to avoid circular import
        self._registry.register_model(name, model_class)
        self._cache[name] = {}
        self._cache_timestamps[name] = {}
    
    def create_model(self, name: str, data: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> Optional['OnyxBaseModel']:
        """Create a model."""
        # Import here to avoid circular import
        model = self._registry.create_model(name, data, id)
        
        if model and model.id:
            self._cache[name][model.id] = model.to_dict()
            self._cache_timestamps[name][model.id] = datetime.utcnow().timestamp()
        
        return model
    
    def get_model(self, name: str, id: str) -> Optional['OnyxBaseModel']:
        """Get a model by ID."""
        # Import here to avoid circular import
        # Check cache
        if name in self._cache and id in self._cache[name]:
            timestamp = self._cache_timestamps[name][id]
            if datetime.utcnow().timestamp() - timestamp <= CACHE_TTL:
                return self._registry.get_model(name, id)
        
        # Get from registry
        model = self._registry.get_model(name, id)
        
        if model:
            self._cache[name][id] = model.to_dict()
            self._cache_timestamps[name][id] = datetime.utcnow().timestamp()
        
        return model
    
    def get_models(self, name: str) -> Dict[str, 'OnyxBaseModel']:
        """Get all models of a type."""
        # Import here to avoid circular import
        models = self._registry.get_models(name)
        
        # Update cache
        for id, model in models.items():
            self._cache[name][id] = model.to_dict()
            self._cache_timestamps[name][id] = datetime.utcnow().timestamp()
        
        return models
    
    def update_model(self, name: str, id: str, data: Dict[str, Any]) -> Optional['OnyxBaseModel']:
        """Update a model."""
        # Import here to avoid circular import
        model = self._registry.update_model(name, id, data)
        
        if model:
            self._cache[name][id] = model.to_dict()
            self._cache_timestamps[name][id] = datetime.utcnow().timestamp()
        
        return model
    
    def delete_model(self, name: str, id: str) -> None:
        """Delete a model."""
        self._registry.delete_model(name, id)
        
        if name in self._cache:
            self._cache[name].pop(id, None)
            self._cache_timestamps[name].pop(id, None)
    
    def clear_models(self, name: str) -> None:
        """Clear all models of a type."""
        self._registry.clear_models(name)
        
        if name in self._cache:
            self._cache[name].clear()
            self._cache_timestamps[name].clear()
    
    def clear_all_models(self) -> None:
        """Clear all models."""
        self._registry.clear_all_models()
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_model_count(self, name: str) -> int:
        """Get count of models of a type."""
        return self._registry.get_model_count(name)
    
    def get_total_model_count(self) -> int:
        """Get total count of all models."""
        return self._registry.get_total_model_count()
    
    def get_model_ids(self, name: str) -> List[str]:
        """Get IDs of all models of a type."""
        return self._registry.get_model_ids(name)
    
    def get_all_model_ids(self) -> Dict[str, List[str]]:
        """Get IDs of all models."""
        return self._registry.get_all_model_ids()
    
    def get_model_data(self, name: str, id: str) -> Optional[Dict[str, Any]]:
        """Get data of a model."""
        # Check cache
        if name in self._cache and id in self._cache[name]:
            timestamp = self._cache_timestamps[name][id]
            if datetime.utcnow().timestamp() - timestamp <= CACHE_TTL:
                return self._cache[name][id]
        
        # Get from registry
        model = self._registry.get_model(name, id)
        
        if model:
            data = model.get_data()
            self._cache[name][id] = data
            self._cache_timestamps[name][id] = datetime.utcnow().timestamp()
            return data
        
        return None
    
    def get_models_data(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Get data of all models of a type."""
        models = self._registry.get_models(name)
        data = {}
        
        for id, model in models.items():
            data[id] = model.get_data()
            self._cache[name][id] = data[id]
            self._cache_timestamps[name][id] = datetime.utcnow().timestamp()
        
        return data
    
    def get_all_models_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get data of all models."""
        return self._registry.get_all_models_data()
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cache."""
        if name:
            if name in self._cache:
                self._cache[name].clear()
                self._cache_timestamps[name].clear()
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    def get_cache_size(self, name: Optional[str] = None) -> int:
        """Get cache size."""
        if name:
            return len(self._cache.get(name, {}))
        return sum(len(cache) for cache in self._cache.values())
    
    def get_cache_hits(self, name: Optional[str] = None) -> int:
        """Get cache hits."""
        # TODO: Implement cache hit tracking
        return 0
    
    def get_cache_misses(self, name: Optional[str] = None) -> int:
        """Get cache misses."""
        # TODO: Implement cache miss tracking
        return 0 