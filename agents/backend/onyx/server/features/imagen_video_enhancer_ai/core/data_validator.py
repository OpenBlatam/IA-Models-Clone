"""
Advanced Data Validator
=======================

Advanced data validation system with schemas and rules.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationRule:
    """Validation rule."""
    field_name: str
    rule_type: str
    validator: Callable[[Any], tuple[bool, Optional[str]]]
    required: bool = True
    level: ValidationLevel = ValidationLevel.MODERATE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSchema:
    """Validation schema."""
    name: str
    rules: List[ValidationRule]
    level: ValidationLevel = ValidationLevel.MODERATE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    validated_data: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add validation error."""
        self.errors.append({
            "field": field,
            "message": message,
            "value": value
        })
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add validation warning."""
        self.warnings.append({
            "field": field,
            "message": message,
            "value": value
        })


class DataValidator:
    """Advanced data validator."""
    
    def __init__(self):
        """Initialize data validator."""
        self.schemas: Dict[str, ValidationSchema] = {}
        self.validators: Dict[str, Callable[[Any], tuple[bool, Optional[str]]]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators."""
        self.validators["required"] = lambda x: (x is not None and x != "", None if (x is not None and x != "") else "Field is required")
        self.validators["string"] = lambda x: (isinstance(x, str), None if isinstance(x, str) else "Must be a string")
        self.validators["integer"] = lambda x: (isinstance(x, int), None if isinstance(x, int) else "Must be an integer")
        self.validators["float"] = lambda x: (isinstance(x, (int, float)), None if isinstance(x, (int, float)) else "Must be a number")
        self.validators["boolean"] = lambda x: (isinstance(x, bool), None if isinstance(x, bool) else "Must be a boolean")
        self.validators["list"] = lambda x: (isinstance(x, list), None if isinstance(x, list) else "Must be a list")
        self.validators["dict"] = lambda x: (isinstance(x, dict), None if isinstance(x, dict) else "Must be a dictionary")
    
    def register_schema(self, schema: ValidationSchema):
        """
        Register a validation schema.
        
        Args:
            schema: Validation schema
        """
        self.schemas[schema.name] = schema
        logger.debug(f"Registered schema: {schema.name}")
    
    def register_validator(self, name: str, validator: Callable[[Any], tuple[bool, Optional[str]]]):
        """
        Register a validator function.
        
        Args:
            name: Validator name
            validator: Validator function
        """
        self.validators[name] = validator
        logger.debug(f"Registered validator: {name}")
    
    def validate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate data against schema.
        
        Args:
            schema_name: Schema name
            data: Data to validate
            level: Optional validation level
            
        Returns:
            Validation result
        """
        if schema_name not in self.schemas:
            result = ValidationResult(is_valid=False)
            result.add_error("schema", f"Schema {schema_name} not found")
            return result
        
        schema = self.schemas[schema_name]
        level = level or schema.level
        result = ValidationResult(is_valid=True)
        result.validated_data = data.copy()
        
        for rule in schema.rules:
            # Skip if rule level doesn't match
            if level == ValidationLevel.STRICT and rule.level != ValidationLevel.STRICT:
                continue
            if level == ValidationLevel.LENIENT and rule.level == ValidationLevel.STRICT:
                continue
            
            field_value = data.get(rule.field_name)
            
            # Check required
            if rule.required and field_value is None:
                result.add_error(rule.field_name, "Field is required")
                continue
            
            # Skip optional fields if None
            if not rule.required and field_value is None:
                continue
            
            # Run validator
            try:
                is_valid, error = rule.validator(field_value)
                if not is_valid:
                    if error:
                        result.add_error(rule.field_name, error, field_value)
                    else:
                        result.add_error(rule.field_name, f"Validation failed for {rule.rule_type}", field_value)
            except Exception as e:
                result.add_error(rule.field_name, f"Validation error: {str(e)}", field_value)
        
        return result
    
    def validate_with_validator(
        self,
        validator_name: str,
        value: Any
    ) -> tuple[bool, Optional[str]]:
        """
        Validate using registered validator.
        
        Args:
            validator_name: Validator name
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if validator_name not in self.validators:
            return False, f"Validator {validator_name} not found"
        
        validator = self.validators[validator_name]
        return validator(value)
    
    def create_schema_builder(self, name: str) -> 'SchemaBuilder':
        """
        Create a schema builder.
        
        Args:
            name: Schema name
            
        Returns:
            Schema builder instance
        """
        return SchemaBuilder(name, self)


class SchemaBuilder:
    """Builder for validation schemas."""
    
    def __init__(self, name: str, validator: DataValidator):
        """
        Initialize schema builder.
        
        Args:
            name: Schema name
            validator: Data validator instance
        """
        self.name = name
        self.validator = validator
        self.rules: List[ValidationRule] = []
        self.level = ValidationLevel.MODERATE
    
    def add_rule(
        self,
        field_name: str,
        rule_type: str,
        validator: Callable[[Any], tuple[bool, Optional[str]]],
        required: bool = True,
        level: ValidationLevel = ValidationLevel.MODERATE
    ) -> 'SchemaBuilder':
        """
        Add validation rule.
        
        Args:
            field_name: Field name
            rule_type: Rule type
            validator: Validator function
            required: Whether field is required
            level: Validation level
            
        Returns:
            Self for chaining
        """
        rule = ValidationRule(
            field_name=field_name,
            rule_type=rule_type,
            validator=validator,
            required=required,
            level=level
        )
        self.rules.append(rule)
        return self
    
    def set_level(self, level: ValidationLevel) -> 'SchemaBuilder':
        """
        Set validation level.
        
        Args:
            level: Validation level
            
        Returns:
            Self for chaining
        """
        self.level = level
        return self
    
    def build(self) -> ValidationSchema:
        """
        Build validation schema.
        
        Returns:
            Validation schema
        """
        schema = ValidationSchema(
            name=self.name,
            rules=self.rules,
            level=self.level
        )
        self.validator.register_schema(schema)
        return schema




