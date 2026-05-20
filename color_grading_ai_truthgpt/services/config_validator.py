"""
Configuration Validator for Color Grading AI
============================================

Advanced configuration validation and schema management.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels."""
    STRICT = "strict"  # All errors are fatal
    WARN = "warn"  # Errors are warnings
    LOOSE = "loose"  # Only critical errors


@dataclass
class ValidationError:
    """Validation error."""
    field: str
    message: str
    level: ValidationLevel
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """
    Configuration validator.
    
    Features:
    - Schema-based validation
    - Type checking
    - Range validation
    - Custom validators
    - Nested validation
    - Error reporting
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        Initialize config validator.
        
        Args:
            validation_level: Validation level
        """
        self.validation_level = validation_level
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._validators: Dict[str, List[Callable]] = {}
    
    def register_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        validators: Optional[List[Callable]] = None
    ):
        """
        Register a validation schema.
        
        Args:
            name: Schema name
            schema: Schema definition
            validators: Optional custom validators
        """
        self._schemas[name] = schema
        if validators:
            self._validators[name] = validators
        logger.debug(f"Registered schema: {name}")
    
    def validate(
        self,
        config: Dict[str, Any],
        schema_name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            schema_name: Optional schema name
            schema: Optional schema definition
            
        Returns:
            Validation result
        """
        if schema_name:
            schema = self._schemas.get(schema_name)
            if not schema:
                return ValidationResult(
                    valid=False,
                    errors=[ValidationError(
                        field="schema",
                        message=f"Schema '{schema_name}' not found",
                        level=ValidationLevel.STRICT
                    )]
                )
        
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="schema",
                    message="No schema provided",
                    level=ValidationLevel.STRICT
                )]
            )
        
        errors = []
        warnings = []
        
        # Validate required fields
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in config:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    level=ValidationLevel.STRICT
                ))
        
        # Validate each field
        for field_name, field_config in schema.get("properties", {}).items():
            if field_name in config:
                field_errors = self._validate_field(
                    field_name,
                    config[field_name],
                    field_config
                )
                errors.extend(field_errors)
            elif field_name in required:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    level=ValidationLevel.STRICT
                ))
        
        # Run custom validators
        if schema_name and schema_name in self._validators:
            for validator in self._validators[schema_name]:
                try:
                    result = validator(config)
                    if result and isinstance(result, ValidationError):
                        errors.append(result)
                    elif result and isinstance(result, str):
                        warnings.append(result)
                except Exception as e:
                    errors.append(ValidationError(
                        field="validator",
                        message=f"Validator error: {str(e)}",
                        level=ValidationLevel.WARN
                    ))
        
        # Separate errors by level
        fatal_errors = [e for e in errors if e.level == ValidationLevel.STRICT]
        warn_errors = [e for e in errors if e.level == ValidationLevel.WARN]
        
        valid = len(fatal_errors) == 0
        
        if self.validation_level == ValidationLevel.WARN:
            warnings.extend([e.message for e in warn_errors])
        elif self.validation_level == ValidationLevel.STRICT:
            fatal_errors.extend(warn_errors)
        
        return ValidationResult(
            valid=valid,
            errors=fatal_errors,
            warnings=warnings
        )
    
    def _validate_field(
        self,
        field_name: str,
        value: Any,
        field_config: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate a single field."""
        errors = []
        
        # Type validation
        expected_type = field_config.get("type")
        if expected_type:
            if expected_type == "string" and not isinstance(value, str):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Expected string, got {type(value).__name__}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected="string"
                ))
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Expected integer, got {type(value).__name__}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected="integer"
                ))
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Expected number, got {type(value).__name__}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected="number"
                ))
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Expected boolean, got {type(value).__name__}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected="boolean"
                ))
        
        # Range validation
        if isinstance(value, (int, float)):
            if "minimum" in field_config and value < field_config["minimum"]:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Value {value} is below minimum {field_config['minimum']}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected=f">= {field_config['minimum']}"
                ))
            
            if "maximum" in field_config and value > field_config["maximum"]:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Value {value} is above maximum {field_config['maximum']}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected=f"<= {field_config['maximum']}"
                ))
        
        # Enum validation
        if "enum" in field_config:
            if value not in field_config["enum"]:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Value '{value}' not in allowed values: {field_config['enum']}",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected=field_config["enum"]
                ))
        
        # Pattern validation (for strings)
        if isinstance(value, str) and "pattern" in field_config:
            import re
            if not re.match(field_config["pattern"], value):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Value '{value}' does not match pattern",
                    level=ValidationLevel.STRICT,
                    value=value,
                    expected=f"pattern: {field_config['pattern']}"
                ))
        
        return errors
    
    def validate_nested(
        self,
        config: Dict[str, Any],
        schema_name: str
    ) -> ValidationResult:
        """
        Validate nested configuration.
        
        Args:
            config: Configuration to validate
            schema_name: Schema name
            
        Returns:
            Validation result
        """
        schema = self._schemas.get(schema_name)
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="schema",
                    message=f"Schema '{schema_name}' not found",
                    level=ValidationLevel.STRICT
                )]
            )
        
        errors = []
        
        # Validate nested objects
        for field_name, field_config in schema.get("properties", {}).items():
            if field_name in config:
                if field_config.get("type") == "object":
                    nested_schema = field_config.get("properties", {})
                    for nested_field, nested_config in nested_schema.items():
                        if nested_field in config[field_name]:
                            nested_errors = self._validate_field(
                                f"{field_name}.{nested_field}",
                                config[field_name][nested_field],
                                nested_config
                            )
                            errors.extend(nested_errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "schemas_count": len(self._schemas),
            "validators_count": sum(len(v) for v in self._validators.values()),
            "validation_level": self.validation_level.value,
        }


