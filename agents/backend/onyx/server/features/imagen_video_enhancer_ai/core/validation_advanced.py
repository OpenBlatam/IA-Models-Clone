"""
Advanced Validation System
==========================

Advanced validation system with schemas, rules, and custom validators.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationRule:
    """Validation rule."""
    field: str
    validator: Callable
    required: bool = True
    error_message: Optional[str] = None
    transform: Optional[Callable] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedValidator:
    """Advanced validator with schema support."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize advanced validator.
        
        Args:
            level: Validation level
        """
        self.level = level
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.validators: Dict[str, Callable] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators."""
        self.validators['required'] = lambda x: x is not None
        self.validators['not_empty'] = lambda x: x is not None and x != ""
        self.validators['is_int'] = lambda x: isinstance(x, int)
        self.validators['is_float'] = lambda x: isinstance(x, (int, float))
        self.validators['is_string'] = lambda x: isinstance(x, str)
        self.validators['is_list'] = lambda x: isinstance(x, list)
        self.validators['is_dict'] = lambda x: isinstance(x, dict)
        self.validators['min_length'] = lambda x, min_len: len(x) >= min_len if hasattr(x, '__len__') else False
        self.validators['max_length'] = lambda x, max_len: len(x) <= max_len if hasattr(x, '__len__') else False
        self.validators['min_value'] = lambda x, min_val: x >= min_val if isinstance(x, (int, float)) else False
        self.validators['max_value'] = lambda x, max_val: x <= max_val if isinstance(x, (int, float)) else False
        self.validators['email'] = lambda x: bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', x)) if isinstance(x, str) else False
        self.validators['url'] = lambda x: bool(re.match(r'^https?://', x)) if isinstance(x, str) else False
    
    def register_validator(self, name: str, validator: Callable):
        """
        Register custom validator.
        
        Args:
            name: Validator name
            validator: Validator function
        """
        self.validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """
        Register validation schema.
        
        Args:
            name: Schema name
            schema: Schema definition
        """
        self.schemas[name] = schema
        logger.info(f"Registered schema: {name}")
    
    def add_rule(self, endpoint: str, rule: ValidationRule):
        """
        Add validation rule.
        
        Args:
            endpoint: Endpoint name
            rule: Validation rule
        """
        if endpoint not in self.rules:
            self.rules[endpoint] = []
        self.rules[endpoint].append(rule)
    
    def validate(
        self,
        data: Dict[str, Any],
        schema_name: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate data.
        
        Args:
            data: Data to validate
            schema_name: Optional schema name
            endpoint: Optional endpoint name
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_data = data.copy()
        
        # Get rules
        rules = []
        if endpoint and endpoint in self.rules:
            rules.extend(self.rules[endpoint])
        if schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
            rules.extend(self._schema_to_rules(schema))
        
        # Apply rules
        for rule in rules:
            field_value = validated_data.get(rule.field)
            
            # Check required
            if rule.required and field_value is None:
                error_msg = rule.error_message or f"Field '{rule.field}' is required"
                if self.level == ValidationLevel.STRICT:
                    errors.append(error_msg)
                elif self.level == ValidationLevel.MODERATE:
                    warnings.append(error_msg)
                continue
            
            if field_value is None:
                continue
            
            # Apply transform
            if rule.transform:
                try:
                    validated_data[rule.field] = rule.transform(field_value)
                except Exception as e:
                    error_msg = f"Transform failed for '{rule.field}': {str(e)}"
                    if self.level == ValidationLevel.STRICT:
                        errors.append(error_msg)
                    else:
                        warnings.append(error_msg)
            
            # Apply validator
            try:
                if not rule.validator(field_value):
                    error_msg = rule.error_message or f"Validation failed for '{rule.field}'"
                    if self.level == ValidationLevel.STRICT:
                        errors.append(error_msg)
                    elif self.level == ValidationLevel.MODERATE:
                        warnings.append(error_msg)
            except Exception as e:
                error_msg = f"Validator error for '{rule.field}': {str(e)}"
                if self.level == ValidationLevel.STRICT:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            data=validated_data
        )
    
    def _schema_to_rules(self, schema: Dict[str, Any]) -> List[ValidationRule]:
        """Convert schema to rules."""
        rules = []
        for field_name, field_def in schema.items():
            if isinstance(field_def, dict):
                validator = field_def.get('validator', lambda x: True)
                required = field_def.get('required', True)
                error_message = field_def.get('error_message')
                transform = field_def.get('transform')
                
                rules.append(ValidationRule(
                    field=field_name,
                    validator=validator,
                    required=required,
                    error_message=error_message,
                    transform=transform
                ))
        return rules



