"""
Validation Framework for Color Grading AI
==========================================

Comprehensive validation framework for data, parameters, and operations.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels."""
    STRICT = "strict"  # All validations must pass
    MODERATE = "moderate"  # Most validations must pass
    LENIENT = "lenient"  # Basic validations only


@dataclass
class ValidationRule:
    """Validation rule."""
    name: str
    validator: Callable
    error_message: str
    required: bool = True
    level: ValidationLevel = ValidationLevel.STRICT


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ValidationFramework:
    """
    Comprehensive validation framework.
    
    Features:
    - Rule-based validation
    - Multiple validation levels
    - Custom validators
    - Parameter validation
    - Data validation
    - Operation validation
    """
    
    def __init__(self, default_level: ValidationLevel = ValidationLevel.STRICT):
        """
        Initialize validation framework.
        
        Args:
            default_level: Default validation level
        """
        self.default_level = default_level
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._validators: Dict[str, Callable] = {}
    
    def register_validator(
        self,
        name: str,
        validator: Callable,
        error_message: str = "Validation failed"
    ):
        """
        Register a validator function.
        
        Args:
            name: Validator name
            validator: Validator function
            error_message: Error message template
        """
        self._validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def add_rule(
        self,
        category: str,
        name: str,
        validator: Union[str, Callable],
        error_message: str = "Validation failed",
        required: bool = True,
        level: Optional[ValidationLevel] = None
    ):
        """
        Add validation rule to a category.
        
        Args:
            category: Rule category
            name: Rule name
            validator: Validator function or name
            error_message: Error message
            required: Whether rule is required
            level: Validation level
        """
        if category not in self._rules:
            self._rules[category] = []
        
        # Get validator function
        if isinstance(validator, str):
            validator_func = self._validators.get(validator)
            if not validator_func:
                raise ValueError(f"Validator not found: {validator}")
        else:
            validator_func = validator
        
        rule = ValidationRule(
            name=name,
            validator=validator_func,
            error_message=error_message,
            required=required,
            level=level or self.default_level
        )
        
        self._rules[category].append(rule)
        logger.info(f"Added validation rule: {category}.{name}")
    
    def validate(
        self,
        category: str,
        data: Any,
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate data against rules in a category.
        
        Args:
            category: Rule category
            data: Data to validate
            level: Validation level
            
        Returns:
            Validation result
        """
        rules = self._rules.get(category, [])
        if not rules:
            return ValidationResult(valid=True)
        
        validation_level = level or self.default_level
        errors = []
        warnings = []
        passed_rules = []
        failed_rules = []
        
        for rule in rules:
            # Skip rules above validation level
            if rule.level.value > validation_level.value:
                continue
            
            try:
                result = rule.validator(data)
                if result:
                    passed_rules.append(rule.name)
                else:
                    if rule.required:
                        errors.append(f"{rule.name}: {rule.error_message}")
                        failed_rules.append(rule.name)
                    else:
                        warnings.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                if rule.required:
                    errors.append(f"{rule.name}: {str(e)}")
                    failed_rules.append(rule.name)
                else:
                    warnings.append(f"{rule.name}: {str(e)}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            passed_rules=passed_rules,
            failed_rules=failed_rules
        )
    
    def validate_color_params(self, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate color grading parameters.
        
        Args:
            params: Color parameters
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Validate brightness
        brightness = params.get("brightness", 0.0)
        if not isinstance(brightness, (int, float)) or brightness < -1.0 or brightness > 1.0:
            errors.append("Brightness must be between -1.0 and 1.0")
        
        # Validate contrast
        contrast = params.get("contrast", 1.0)
        if not isinstance(contrast, (int, float)) or contrast < 0.0 or contrast > 3.0:
            errors.append("Contrast must be between 0.0 and 3.0")
        
        # Validate saturation
        saturation = params.get("saturation", 1.0)
        if not isinstance(saturation, (int, float)) or saturation < 0.0 or saturation > 2.0:
            errors.append("Saturation must be between 0.0 and 2.0")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_rules = sum(len(rules) for rules in self._rules.values())
        return {
            "total_validators": len(self._validators),
            "total_rules": total_rules,
            "categories": list(self._rules.keys()),
            "default_level": self.default_level.value,
        }


