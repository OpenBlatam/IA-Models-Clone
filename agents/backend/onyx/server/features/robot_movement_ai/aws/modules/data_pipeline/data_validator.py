"""
Data Validator
==============

Data validation utilities.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error."""
    field: str
    message: str
    value: Any = None


class DataValidator:
    """Data validator."""
    
    def __init__(self):
        self._validators: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_validator(self, name: str, validator: Callable):
        """Register validator."""
        self._validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register validation schema."""
        self._schemas[name] = schema
        logger.info(f"Registered schema: {name}")
    
    def validate(self, data: Any, schema_name: str) -> Tuple[bool, List[ValidationError]]:
        """Validate data against schema."""
        if schema_name not in self._schemas:
            raise ValueError(f"Schema {schema_name} not found")
        
        schema = self._schemas[schema_name]
        errors = []
        
        if isinstance(data, dict):
            errors.extend(self._validate_dict(data, schema))
        else:
            errors.extend(self._validate_value(data, schema))
        
        return len(errors) == 0, errors
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate dictionary."""
        errors = []
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing"
                ))
        
        # Validate each field
        for field, value in data.items():
            if field in schema.get("properties", {}):
                field_schema = schema["properties"][field]
                errors.extend(self._validate_value(value, field_schema, field))
        
        return errors
    
    def _validate_value(self, value: Any, schema: Dict[str, Any], field: str = "") -> List[ValidationError]:
        """Validate single value."""
        errors = []
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if expected_type == "string" and not isinstance(value, str):
                errors.append(ValidationError(
                    field=field,
                    message=f"Expected string, got {type(value).__name__}",
                    value=value
                ))
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(ValidationError(
                    field=field,
                    message=f"Expected integer, got {type(value).__name__}",
                    value=value
                ))
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(ValidationError(
                    field=field,
                    message=f"Expected number, got {type(value).__name__}",
                    value=value
                ))
        
        # Pattern validation
        pattern = schema.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                errors.append(ValidationError(
                    field=field,
                    message=f"Value does not match pattern: {pattern}",
                    value=value
                ))
        
        # Range validation
        if isinstance(value, (int, float)):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            
            if minimum is not None and value < minimum:
                errors.append(ValidationError(
                    field=field,
                    message=f"Value {value} is less than minimum {minimum}",
                    value=value
                ))
            
            if maximum is not None and value > maximum:
                errors.append(ValidationError(
                    field=field,
                    message=f"Value {value} is greater than maximum {maximum}",
                    value=value
                ))
        
        # Enum validation
        enum_values = schema.get("enum")
        if enum_values and value not in enum_values:
            errors.append(ValidationError(
                field=field,
                message=f"Value {value} is not in allowed values: {enum_values}",
                value=value
            ))
        
        return errors
    
    def validate_with_validator(self, data: Any, validator_name: str) -> Tuple[bool, Optional[str]]:
        """Validate using named validator."""
        if validator_name not in self._validators:
            raise ValueError(f"Validator {validator_name} not found")
        
        validator = self._validators[validator_name]
        try:
            result = validator(data)
            if isinstance(result, bool):
                return result, None if result else "Validation failed"
            elif isinstance(result, tuple):
                return result
            else:
                return True, None
        except Exception as e:
            return False, str(e)

