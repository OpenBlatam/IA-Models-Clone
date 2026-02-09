"""
Schema Validator for Flux2 Clothing Changer
===========================================

Advanced schema validation system.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SchemaField:
    """Schema field definition."""
    name: str
    field_type: type
    required: bool = True
    default: Any = None
    validators: List[Callable] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.validators is None:
            self.validators = []


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SchemaValidator:
    """Advanced schema validation system."""
    
    def __init__(self):
        """Initialize schema validator."""
        self.schemas: Dict[str, List[SchemaField]] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_schema(
        self,
        schema_name: str,
        fields: List[SchemaField],
    ) -> None:
        """
        Register validation schema.
        
        Args:
            schema_name: Schema name
            fields: List of schema fields
        """
        self.schemas[schema_name] = fields
        logger.info(f"Registered schema: {schema_name}")
    
    def register_validator(
        self,
        validator_name: str,
        validator: Callable[[Any], Tuple[bool, Optional[str]]],
    ) -> None:
        """
        Register custom validator.
        
        Args:
            validator_name: Validator name
            validator: Validator function
        """
        self.custom_validators[validator_name] = validator
        logger.info(f"Registered validator: {validator_name}")
    
    def validate(
        self,
        schema_name: str,
        data: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate data against schema.
        
        Args:
            schema_name: Schema name
            data: Data to validate
            
        Returns:
            Validation result
        """
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema not found: {schema_name}"],
            )
        
        fields = self.schemas[schema_name]
        errors = []
        warnings = []
        
        for field in fields:
            field_name = field.name
            field_value = data.get(field_name, field.default)
            
            # Check required
            if field.required and field_value is None:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            # Check type
            if field_value is not None and not isinstance(field_value, field.field_type):
                errors.append(
                    f"Field '{field_name}' expected type {field.field_type.__name__}, "
                    f"got {type(field_value).__name__}"
                )
                continue
            
            # Run validators
            for validator in field.validators:
                try:
                    is_valid, error_message = validator(field_value)
                    if not is_valid:
                        if error_message:
                            errors.append(f"Field '{field_name}': {error_message}")
                        else:
                            errors.append(f"Field '{field_name}' validation failed")
                except Exception as e:
                    errors.append(f"Field '{field_name}' validator error: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "total_schemas": len(self.schemas),
            "schemas": list(self.schemas.keys()),
            "custom_validators": list(self.custom_validators.keys()),
        }


