"""
Validation Framework for Color Grading AI
==========================================

Unified validation framework with rules, validators, and error handling.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import InvalidParametersError

logger = logging.getLogger(__name__)


class ValidationRuleType(Enum):
    """Validation rule types."""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    ENUM = "enum"
    LENGTH = "length"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    rule_type: ValidationRuleType
    field_name: str
    validator: Optional[Callable] = None
    error_message: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class ValidationFramework:
    """
    Unified validation framework.
    
    Features:
    - Rule-based validation
    - Custom validators
    - Error aggregation
    - Type checking
    - Range validation
    - Pattern matching
    """
    
    def __init__(self):
        """Initialize validation framework."""
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._validators: Dict[str, Callable] = {}
    
    def register_validator(self, name: str, validator: Callable):
        """
        Register custom validator.
        
        Args:
            name: Validator name
            validator: Validator function
        """
        self._validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def add_rule(
        self,
        schema_name: str,
        field_name: str,
        rule_type: ValidationRuleType,
        validator: Optional[Callable] = None,
        error_message: Optional[str] = None,
        **params
    ):
        """
        Add validation rule.
        
        Args:
            schema_name: Schema name
            field_name: Field name
            rule_type: Rule type
            validator: Optional validator function
            error_message: Optional error message
            **params: Rule parameters
        """
        if schema_name not in self._rules:
            self._rules[schema_name] = []
        
        rule = ValidationRule(
            rule_type=rule_type,
            field_name=field_name,
            validator=validator,
            error_message=error_message,
            params=params
        )
        
        self._rules[schema_name].append(rule)
        logger.debug(f"Added rule for {schema_name}.{field_name}")
    
    def validate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            schema_name: Schema name
            data: Data to validate
            raise_on_error: Whether to raise on error
            
        Returns:
            Validated data
            
        Raises:
            InvalidParametersError: If validation fails
        """
        rules = self._rules.get(schema_name, [])
        errors = []
        validated = {}
        
        for rule in rules:
            field_name = rule.field_name
            value = data.get(field_name)
            
            try:
                # Apply rule
                if rule.rule_type == ValidationRuleType.REQUIRED:
                    if value is None:
                        errors.append(f"{field_name} is required")
                        continue
                
                elif rule.rule_type == ValidationRuleType.TYPE:
                    expected_type = rule.params.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(f"{field_name} must be {expected_type.__name__}")
                        continue
                
                elif rule.rule_type == ValidationRuleType.RANGE:
                    min_val = rule.params.get("min")
                    max_val = rule.params.get("max")
                    if min_val is not None and value < min_val:
                        errors.append(f"{field_name} must be >= {min_val}")
                        continue
                    if max_val is not None and value > max_val:
                        errors.append(f"{field_name} must be <= {max_val}")
                        continue
                
                elif rule.rule_type == ValidationRuleType.PATTERN:
                    pattern = rule.params.get("pattern")
                    import re
                    if pattern and not re.match(pattern, str(value)):
                        errors.append(f"{field_name} does not match pattern")
                        continue
                
                elif rule.rule_type == ValidationRuleType.ENUM:
                    allowed = rule.params.get("values", [])
                    if value not in allowed:
                        errors.append(f"{field_name} must be one of {allowed}")
                        continue
                
                elif rule.rule_type == ValidationRuleType.LENGTH:
                    min_len = rule.params.get("min_length")
                    max_len = rule.params.get("max_length")
                    if min_len is not None and len(value) < min_len:
                        errors.append(f"{field_name} length must be >= {min_len}")
                        continue
                    if max_len is not None and len(value) > max_len:
                        errors.append(f"{field_name} length must be <= {max_len}")
                        continue
                
                elif rule.rule_type == ValidationRuleType.CUSTOM:
                    if rule.validator:
                        if not rule.validator(value, data):
                            error_msg = rule.error_message or f"{field_name} validation failed"
                            errors.append(error_msg)
                            continue
                
                # Value passed validation
                validated[field_name] = value
            
            except Exception as e:
                errors.append(f"Error validating {field_name}: {e}")
        
        # Check for errors
        if errors:
            error_message = "; ".join(errors)
            if raise_on_error:
                raise InvalidParametersError(error_message)
            else:
                logger.warning(f"Validation errors: {error_message}")
        
        return validated
    
    def create_schema(
        self,
        schema_name: str,
        rules: List[Dict[str, Any]]
    ):
        """
        Create validation schema from rules.
        
        Args:
            schema_name: Schema name
            rules: List of rule definitions
        """
        for rule_def in rules:
            self.add_rule(schema_name, **rule_def)


# Predefined schemas
def create_color_params_schema(framework: ValidationFramework):
    """Create color parameters validation schema."""
    framework.create_schema("color_params", [
        {
            "field_name": "brightness",
            "rule_type": ValidationRuleType.RANGE,
            "params": {"min": -1.0, "max": 1.0},
            "error_message": "Brightness must be between -1.0 and 1.0"
        },
        {
            "field_name": "contrast",
            "rule_type": ValidationRuleType.RANGE,
            "params": {"min": 0.0, "max": 3.0},
            "error_message": "Contrast must be between 0.0 and 3.0"
        },
        {
            "field_name": "saturation",
            "rule_type": ValidationRuleType.RANGE,
            "params": {"min": 0.0, "max": 3.0},
            "error_message": "Saturation must be between 0.0 and 3.0"
        },
    ])




