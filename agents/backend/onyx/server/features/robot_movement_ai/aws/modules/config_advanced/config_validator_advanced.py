"""
Advanced Config Validator
=========================

Advanced configuration validation.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule."""
    key: str
    validator: Callable
    required: bool = False
    default: Any = None
    description: Optional[str] = None


@dataclass
class ValidationError:
    """Validation error."""
    key: str
    message: str
    value: Any = None


class AdvancedConfigValidator:
    """Advanced configuration validator."""
    
    def __init__(self):
        self._rules: Dict[str, ValidationRule] = {}
        self._custom_validators: Dict[str, Callable] = {}
    
    def add_rule(
        self,
        key: str,
        validator: Callable,
        required: bool = False,
        default: Any = None,
        description: Optional[str] = None
    ):
        """Add validation rule."""
        rule = ValidationRule(
            key=key,
            validator=validator,
            required=required,
            default=default,
            description=description
        )
        
        self._rules[key] = rule
        logger.info(f"Added validation rule for: {key}")
    
    def register_validator(self, name: str, validator: Callable):
        """Register custom validator."""
        self._custom_validators[name] = validator
        logger.info(f"Registered custom validator: {name}")
    
    def validate(self, config: Dict[str, Any]) -> tuple[bool, List[ValidationError]]:
        """Validate configuration."""
        errors = []
        
        # Check required fields
        for key, rule in self._rules.items():
            if rule.required and key not in config:
                errors.append(ValidationError(
                    key=key,
                    message=f"Required field '{key}' is missing"
                ))
        
        # Validate values
        for key, value in config.items():
            if key in self._rules:
                rule = self._rules[key]
                
                try:
                    if not rule.validator(value):
                        errors.append(ValidationError(
                            key=key,
                            message=f"Validation failed for '{key}'",
                            value=value
                        ))
                except Exception as e:
                    errors.append(ValidationError(
                        key=key,
                        message=f"Validation error for '{key}': {str(e)}",
                        value=value
                    ))
        
        return len(errors) == 0, errors
    
    def validate_and_apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply defaults."""
        validated = config.copy()
        
        for key, rule in self._rules.items():
            if key not in validated:
                if rule.default is not None:
                    validated[key] = rule.default
                elif rule.required:
                    raise ValueError(f"Required field '{key}' is missing")
        
        is_valid, errors = self.validate(validated)
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")
        
        return validated
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_rules": len(self._rules),
            "required_fields": sum(1 for r in self._rules.values() if r.required),
            "custom_validators": len(self._custom_validators)
        }















