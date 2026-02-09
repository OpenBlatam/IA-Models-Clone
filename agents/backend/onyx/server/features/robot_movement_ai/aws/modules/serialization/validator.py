"""
Schema Validator
================

Schema validation for data.
"""

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Schema validator."""
    
    def __init__(self):
        self._schemas: Dict[str, type[BaseModel]] = {}
    
    def register_schema(self, name: str, schema: type[BaseModel]):
        """Register schema."""
        self._schemas[name] = schema
        logger.debug(f"Registered schema: {name}")
    
    def validate(self, name: str, data: Any) -> BaseModel:
        """Validate data against schema."""
        if name not in self._schemas:
            raise ValueError(f"Schema {name} not found")
        
        schema = self._schemas[name]
        try:
            return schema(**data) if isinstance(data, dict) else schema.parse_obj(data)
        except ValidationError as e:
            logger.error(f"Validation failed for {name}: {e}")
            raise
    
    def validate_dict(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dict and return validated dict."""
        validated = self.validate(name, data)
        return validated.dict()















