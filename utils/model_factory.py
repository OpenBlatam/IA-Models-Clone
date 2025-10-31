from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import CACHE_TTL, VALIDATION_TIMEOUT
from .model_field import ModelField, FieldConfig
from .model_schema import ModelSchema, SchemaConfig
        from .base_model import OnyxBaseModel
        from .base_model import OnyxBaseModel
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Factory - Onyx Integration
Factory for creating models with schemas.
"""
# from .base_model import OnyxBaseModel  # REMOVED to break circular import

T = TypeVar('T')

class ModelFactory:
    """Factory for creating models with schemas."""
    
    def __init__(self) -> Any:
        """Initialize factory."""
        self._schemas: Dict[str, ModelSchema] = {}
        self._models: Dict[str, Type["OnyxBaseModel"]] = {}
    
    def register_schema(self, name: str, schema: ModelSchema) -> None:
        """Register a schema."""
        if name in self._schemas:
            raise ValueError(f"Schema {name} already exists")
        
        self._schemas[name] = schema
    
    def register_model(self, name: str, model_class: Type["OnyxBaseModel"]) -> None:
        """Register a model class."""
        if name in self._models:
            raise ValueError(f"Model {name} already exists")
        
        self._models[name] = model_class
    
    def get_schema(self, name: str) -> Optional[ModelSchema]:
        """Get a schema."""
        return self._schemas.get(name)
    
    def get_model_class(self, name: str) -> Optional[Type["OnyxBaseModel"]]:
        return self._models.get(name)
    
    def create_model(self, name: str, data: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> Optional["OnyxBaseModel"]:
        """Create a model."""
        schema = self.get_schema(name)
        model_class = self.get_model_class(name)
        
        if not schema or not model_class:
            return None
        
        return model_class(schema=schema, data=data, id=id)
    
    def create_field(self, field_type: Type[T], config: Optional[FieldConfig] = None, name: Optional[str] = None) -> ModelField:
        """Create a field."""
        return ModelField(field_type=field_type, config=config, name=name)
    
    def create_schema(self, fields: Dict[str, ModelField], config: Optional[SchemaConfig] = None, name: Optional[str] = None) -> ModelSchema:
        """Create a schema."""
        return ModelSchema(fields=fields, config=config, name=name)
    
    def create_string_field(self, required: bool = False, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, choices: Optional[List[str]] = None, format_type: Optional[str] = None, name: Optional[str] = None) -> ModelField:
        """Create a string field."""
        config = FieldConfig(
            required=required,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            choices=choices,
            format_type=format_type
        )
        return self.create_field(str, config=config, name=name)
    
    def create_integer_field(self, required: bool = False, min_value: Optional[int] = None, max_value: Optional[int] = None, choices: Optional[List[int]] = None, name: Optional[str] = None) -> ModelField:
        """Create an integer field."""
        config = FieldConfig(
            required=required,
            min_value=min_value,
            max_value=max_value,
            choices=choices
        )
        return self.create_field(int, config=config, name=name)
    
    def create_float_field(self, required: bool = False, min_value: Optional[float] = None, max_value: Optional[float] = None, choices: Optional[List[float]] = None, name: Optional[str] = None) -> ModelField:
        """Create a float field."""
        config = FieldConfig(
            required=required,
            min_value=min_value,
            max_value=max_value,
            choices=choices
        )
        return self.create_field(float, config=config, name=name)
    
    def create_boolean_field(self, required: bool = False, default: Optional[bool] = None, name: Optional[str] = None) -> ModelField:
        """Create a boolean field."""
        config = FieldConfig(
            required=required,
            default=default
        )
        return self.create_field(bool, config=config, name=name)
    
    def create_datetime_field(self, required: bool = False, default: Optional[datetime] = None, name: Optional[str] = None) -> ModelField:
        """Create a datetime field."""
        config = FieldConfig(
            required=required,
            default=default
        )
        return self.create_field(datetime, config=config, name=name)
    
    def create_list_field(self, item_type: Type[T], required: bool = False, default: Optional[List[T]] = None, name: Optional[str] = None) -> ModelField:
        """Create a list field."""
        config = FieldConfig(
            required=required,
            default=default
        )
        return self.create_field(List[item_type], config=config, name=name)
    
    def create_dict_field(self, required: bool = False, default: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> ModelField:
        """Create a dictionary field."""
        config = FieldConfig(
            required=required,
            default=default
        )
        return self.create_field(Dict[str, Any], config=config, name=name) 