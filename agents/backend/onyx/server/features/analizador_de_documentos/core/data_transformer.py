"""
Data Transformer for Document Analyzer
=======================================

Advanced data transformation utilities.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)

class DataTransformer:
    """Advanced data transformer"""
    
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
        logger.info("DataTransformer initialized")
    
    def register_transformer(self, name: str, transformer: Callable):
        """Register a transformer function"""
        self.transformers[name] = transformer
        logger.info(f"Registered transformer: {name}")
    
    def transform(self, data: Any, transformer_name: str, **kwargs) -> Any:
        """Transform data using registered transformer"""
        if transformer_name not in self.transformers:
            raise ValueError(f"Transformer {transformer_name} not found")
        
        transformer = self.transformers[transformer_name]
        return transformer(data, **kwargs)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def extract_fields(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Extract specific fields from data"""
        return {field: data.get(field) for field in fields if field in data}
    
    def flatten_dict(self, data: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(self.flatten_dict(value, separator, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    def unflatten_dict(self, data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Unflatten dictionary"""
        result = {}
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result
    
    def convert_datetime(self, data: Any, format: str = 'iso') -> str:
        """Convert datetime to string"""
        if isinstance(data, datetime):
            if format == 'iso':
                return data.isoformat()
            elif format == 'timestamp':
                return str(data.timestamp())
            else:
                return data.strftime(format)
        return str(data)
    
    def sanitize_dict(self, data: Dict[str, Any], remove_none: bool = True) -> Dict[str, Any]:
        """Sanitize dictionary"""
        result = {}
        for key, value in data.items():
            if remove_none and value is None:
                continue
            if isinstance(value, dict):
                result[key] = self.sanitize_dict(value, remove_none)
            elif isinstance(value, list):
                result[key] = [self.sanitize_dict(item, remove_none) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result

# Global instance
data_transformer = DataTransformer()
















