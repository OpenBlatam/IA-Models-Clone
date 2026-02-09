"""
Data Transformer
================

Data transformation utilities.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import json

logger = logging.getLogger(__name__)


class DataTransformer:
    """Data transformation utilities."""
    
    def __init__(self):
        self._transformers: Dict[str, Callable] = {}
    
    def register_transformer(self, name: str, transformer: Callable):
        """Register transformer."""
        self._transformers[name] = transformer
        logger.info(f"Registered transformer: {name}")
    
    def transform(self, data: Any, transformer_name: str, **kwargs) -> Any:
        """Transform data using named transformer."""
        if transformer_name not in self._transformers:
            raise ValueError(f"Transformer {transformer_name} not found")
        
        transformer = self._transformers[transformer_name]
        return transformer(data, **kwargs)
    
    def normalize(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data according to schema."""
        normalized = {}
        
        for key, value in schema.items():
            if key in data:
                normalized[key] = self._normalize_value(data[key], value)
            elif "default" in value:
                normalized[key] = value["default"]
        
        return normalized
    
    def _normalize_value(self, value: Any, schema: Dict[str, Any]) -> Any:
        """Normalize single value."""
        value_type = schema.get("type", "string")
        
        if value_type == "integer":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "boolean":
            return bool(value)
        elif value_type == "string":
            return str(value)
        elif value_type == "array":
            return list(value) if isinstance(value, (list, tuple)) else [value]
        else:
            return value
    
    def flatten(self, data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        flattened = {}
        
        def _flatten(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}{separator}{key}" if prefix else key
                    _flatten(value, new_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}{separator}{i}" if prefix else str(i)
                    _flatten(item, new_key)
            else:
                flattened[prefix] = obj
        
        _flatten(data)
        return flattened
    
    def unflatten(self, data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Unflatten dictionary."""
        unflattened = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = unflattened
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return unflattened
    
    def to_json(self, data: Any) -> str:
        """Convert data to JSON."""
        return json.dumps(data, default=str)
    
    def from_json(self, json_str: str) -> Any:
        """Parse JSON string."""
        return json.loads(json_str)















