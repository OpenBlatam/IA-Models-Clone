from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from .base_types import CACHE_TTL, VALIDATION_TIMEOUT
from .model_field import ModelField, FieldConfig
from .model_schema import ModelSchema, SchemaConfig
from .model_factory import ModelFactory
from .model_registry import ModelRegistry
from .model_manager import ModelManager
from .model_service import ModelService
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Repository - Onyx Integration
Repository for persisting models.
"""
# from .base_model import OnyxBaseModel  # REMOVED to break circular import

T = TypeVar('T')

class ModelRepository:
    """Repository for persisting models."""
    
    def __init__(self, base_path: str = "data"):
        """Initialize repository."""
        self._service = ModelService()
        self._base_path = base_path
        self._ensure_base_path()
    
    def _ensure_base_path(self) -> None:
        """Ensure base path exists."""
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)
    
    def _get_model_path(self, name: str) -> str:
        """Get path for model data."""
        return os.path.join(self._base_path, f"{name}.json")
    
    def _load_models(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Load models from file."""
        path = self._get_model_path(name)
        
        if not os.path.exists(path):
            return {}
        
        try:
            with open(path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        except Exception as e:
            print(f"Error loading models from {path}: {e}")
            return {}
    
    def _save_models(self, name: str, models: Dict[str, Dict[str, Any]]) -> None:
        """Save models to file."""
        path = self._get_model_path(name)
        
        try:
            with open(path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(models, f, indent=2)
        except Exception as e:
            print(f"Error saving models to {path}: {e}")
    
    def register_schema(self, name: str, schema: ModelSchema) -> None:
        """Register a schema."""
        self._service.register_schema(name, schema)
    
    def register_model(self, name: str, model_class: "OnyxBaseModel") -> None:
        self._service.register_model(name, model_class)
        # Load existing models
        models_data = self._load_models(name)
        for id, data in models_data.items():
            self._service.create_model(name, data, id)
    
    def create_model(self, name: str, data: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> Optional["OnyxBaseModel"]:
        model = self._service.create_model(name, data, id)
        if model and model.id:
            # Save to file
            models_data = self._load_models(name)
            models_data[model.id] = model.get_data()
            self._save_models(name, models_data)
        return model
    
    def get_model(self, name: str, id: str) -> Optional["OnyxBaseModel"]:
        return self._service.get_model(name, id)
    
    def get_models(self, name: str) -> Dict[str, "OnyxBaseModel"]:
        return self._service.get_models(name)
    
    def update_model(self, name: str, id: str, data: Dict[str, Any]) -> Optional["OnyxBaseModel"]:
        model = self._service.update_model(name, id, data)
        if model:
            # Save to file
            models_data = self._load_models(name)
            models_data[model.id] = model.get_data()
            self._save_models(name, models_data)
        return model
    
    def delete_model(self, name: str, id: str) -> None:
        self._service.delete_model(name, id)
        # Update file
        models_data = self._load_models(name)
        models_data.pop(id, None)
        self._save_models(name, models_data)
    
    def clear_models(self, name: str) -> None:
        self._service.clear_models(name)
        # Clear file
        self._save_models(name, {})
    
    def clear_all_models(self) -> None:
        self._service.clear_all_models()
        # Clear all files
        for name in self._service.get_all_model_ids().keys():
            self._save_models(name, {})
    
    def get_model_count(self, name: str) -> int:
        return self._service.get_model_count(name)
    
    def get_total_model_count(self) -> int:
        return self._service.get_total_model_count()
    
    def get_model_ids(self, name: str) -> List[str]:
        return self._service.get_model_ids(name)
    
    def get_all_model_ids(self) -> Dict[str, List[str]]:
        return self._service.get_all_model_ids()
    
    def get_model_data(self, name: str, id: str) -> Optional[Dict[str, Any]]:
        return self._service.get_model_data(name, id)
    
    def get_models_data(self, name: str) -> Dict[str, Dict[str, Any]]:
        return self._service.get_models_data(name)
    
    def get_all_models_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self._service.get_all_models_data()
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        self._service.clear_cache(name)
    
    def get_cache_size(self, name: Optional[str] = None) -> int:
        return self._service.get_cache_size(name)
    
    def get_validation_cache_size(self, name: Optional[str] = None) -> int:
        return self._service.get_validation_cache_size(name)
    
    def get_cache_hits(self, name: Optional[str] = None) -> int:
        return self._service.get_cache_hits(name)
    
    def get_cache_misses(self, name: Optional[str] = None) -> int:
        return self._service.get_cache_misses(name)
    
    def get_validation_hits(self, name: Optional[str] = None) -> int:
        return self._service.get_validation_hits(name)
    
    def get_validation_misses(self, name: Optional[str] = None) -> int:
        return self._service.get_validation_misses(name) 