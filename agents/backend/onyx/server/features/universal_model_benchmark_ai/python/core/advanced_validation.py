"""
Advanced Validation Module - Comprehensive validation system.

Provides:
- Schema validation
- Data sanitization
- Custom validators
- Validation chains
- Error aggregation
"""

import logging
import re
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationError:
    """Validation error."""
    field: str
    message: str
    value: Any = None
    code: str = "validation_error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "value": self.value,
            "code": self.code,
        }


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, value: Any = None, code: str = "validation_error") -> None:
        """Add validation error."""
        self.errors.append(ValidationError(field, message, value, code))
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
        }


class Validator:
    """Base validator class."""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate value.
        
        Args:
            value: Value to validate
            context: Optional context
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        self._validate(value, result, context or {})
        return result
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        """Override in subclasses."""
        pass


class RequiredValidator(Validator):
    """Required field validator."""
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is None or value == "":
            result.add_error("field", "Field is required", value, "required")


class TypeValidator(Validator):
    """Type validator."""
    
    def __init__(self, expected_type: type):
        """Initialize type validator."""
        self.expected_type = expected_type
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is not None and not isinstance(value, self.expected_type):
            result.add_error(
                "field",
                f"Expected type {self.expected_type.__name__}, got {type(value).__name__}",
                value,
                "type_mismatch"
            )


class RangeValidator(Validator):
    """Range validator."""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        """Initialize range validator."""
        self.min_value = min_value
        self.max_value = max_value
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is None:
            return
        
        try:
            num_value = float(value)
            if self.min_value is not None and num_value < self.min_value:
                result.add_error(
                    "field",
                    f"Value must be >= {self.min_value}",
                    value,
                    "min_value"
                )
            if self.max_value is not None and num_value > self.max_value:
                result.add_error(
                    "field",
                    f"Value must be <= {self.max_value}",
                    value,
                    "max_value"
                )
        except (ValueError, TypeError):
            result.add_error("field", "Value must be numeric", value, "not_numeric")


class RegexValidator(Validator):
    """Regex validator."""
    
    def __init__(self, pattern: str, message: str = "Value does not match pattern"):
        """Initialize regex validator."""
        self.pattern = re.compile(pattern)
        self.message = message
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is None:
            return
        
        if not self.pattern.match(str(value)):
            result.add_error("field", self.message, value, "pattern_mismatch")


class LengthValidator(Validator):
    """Length validator."""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        """Initialize length validator."""
        self.min_length = min_length
        self.max_length = max_length
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is None:
            return
        
        length = len(str(value))
        if self.min_length is not None and length < self.min_length:
            result.add_error(
                "field",
                f"Length must be >= {self.min_length}",
                value,
                "min_length"
            )
        if self.max_length is not None and length > self.max_length:
            result.add_error(
                "field",
                f"Length must be <= {self.max_length}",
                value,
                "max_length"
            )


class CustomValidator(Validator):
    """Custom validator."""
    
    def __init__(self, validator_func: Callable[[Any], Union[bool, str]]):
        """
        Initialize custom validator.
        
        Args:
            validator_func: Function that returns True/False or error message
        """
        self.validator_func = validator_func
    
    def _validate(self, value: Any, result: ValidationResult, context: Dict[str, Any]) -> None:
        if value is None:
            return
        
        validation_result = self.validator_func(value)
        if isinstance(validation_result, bool):
            if not validation_result:
                result.add_error("field", "Custom validation failed", value, "custom_validation")
        elif isinstance(validation_result, str):
            result.add_error("field", validation_result, value, "custom_validation")


class ValidationSchema:
    """Validation schema."""
    
    def __init__(self):
        """Initialize validation schema."""
        self.fields: Dict[str, List[Validator]] = {}
    
    def add_field(self, field_name: str, *validators: Validator) -> None:
        """
        Add field with validators.
        
        Args:
            field_name: Field name
            *validators: Validator instances
        """
        if field_name not in self.fields:
            self.fields[field_name] = []
        self.fields[field_name].extend(validators)
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        
        for field_name, validators in self.fields.items():
            value = data.get(field_name)
            
            for validator in validators:
                field_result = validator.validate(value, {"field": field_name, "data": data})
                
                if not field_result.valid:
                    for error in field_result.errors:
                        error.field = field_name
                        result.errors.append(error)
                        result.valid = False
                
                result.warnings.extend(field_result.warnings)
        
        return result


class AdvancedValidator:
    """Advanced validation manager."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize advanced validator.
        
        Args:
            level: Validation level
        """
        self.level = level
        self.schemas: Dict[str, ValidationSchema] = {}
    
    def create_schema(self, name: str) -> ValidationSchema:
        """
        Create validation schema.
        
        Args:
            name: Schema name
            
        Returns:
            Validation schema
        """
        schema = ValidationSchema()
        self.schemas[name] = schema
        return schema
    
    def validate(self, schema_name: str, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data using schema.
        
        Args:
            schema_name: Schema name
            data: Data to validate
            
        Returns:
            Validation result
        """
        if schema_name not in self.schemas:
            result = ValidationResult(valid=False)
            result.add_error("schema", f"Schema {schema_name} not found")
            return result
        
        return self.schemas[schema_name].validate(data)












