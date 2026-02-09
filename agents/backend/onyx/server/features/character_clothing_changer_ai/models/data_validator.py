"""
Data Validator for Flux2 Clothing Changer
=========================================

Advanced data validation system.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule."""
    field_name: str
    rule_type: str
    validator: Callable[[Any], Tuple[bool, Optional[str]]]
    required: bool = True
    default_value: Any = None


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    validated_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.validated_data is None:
            self.validated_data = {}


class DataValidator:
    """Advanced data validation system."""
    
    def __init__(self):
        """Initialize data validator."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_schema(
        self,
        schema_name: str,
        rules: List[ValidationRule],
    ) -> None:
        """
        Register validation schema.
        
        Args:
            schema_name: Schema name
            rules: List of validation rules
        """
        self.rules[schema_name] = rules
        logger.info(f"Registered validation schema: {schema_name}")
    
    def register_custom_validator(
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
        logger.info(f"Registered custom validator: {validator_name}")
    
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
        if schema_name not in self.rules:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema not found: {schema_name}"],
            )
        
        rules = self.rules[schema_name]
        errors = []
        warnings = []
        validated_data = {}
        
        for rule in rules:
            field_name = rule.field_name
            field_value = data.get(field_name, rule.default_value)
            
            # Check required
            if rule.required and field_value is None:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            # Apply default if not provided
            if field_value is None and rule.default_value is not None:
                field_value = rule.default_value
            
            # Validate
            is_valid, error_message = rule.validator(field_value)
            
            if not is_valid:
                if error_message:
                    errors.append(f"Field '{field_name}': {error_message}")
                else:
                    errors.append(f"Field '{field_name}' validation failed")
            else:
                validated_data[field_name] = field_value
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_data=validated_data,
        )
    
    @staticmethod
    def create_type_validator(expected_type: type) -> Callable:
        """Create type validator."""
        def validator(value: Any) -> Tuple[bool, Optional[str]]:
            if not isinstance(value, expected_type):
                return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
            return True, None
        return validator
    
    @staticmethod
    def create_range_validator(min_value: float, max_value: float) -> Callable:
        """Create range validator."""
        def validator(value: Any) -> Tuple[bool, Optional[str]]:
            try:
                num_value = float(value)
                if not (min_value <= num_value <= max_value):
                    return False, f"Value {num_value} not in range [{min_value}, {max_value}]"
                return True, None
            except (ValueError, TypeError):
                return False, "Value is not a number"
        return validator
    
    @staticmethod
    def create_length_validator(min_length: int, max_length: int) -> Callable:
        """Create length validator."""
        def validator(value: Any) -> Tuple[bool, Optional[str]]:
            if not hasattr(value, "__len__"):
                return False, "Value has no length"
            length = len(value)
            if not (min_length <= length <= max_length):
                return False, f"Length {length} not in range [{min_length}, {max_length}]"
            return True, None
        return validator
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "total_schemas": len(self.rules),
            "schemas": list(self.rules.keys()),
            "custom_validators": list(self.custom_validators.keys()),
        }


