"""
Validation Manager
==================

Centralized validation system.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
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
    name: str
    validator: Callable[[Any], tuple[bool, Optional[str]]]
    level: ValidationLevel = ValidationLevel.MODERATE
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class ValidationManager:
    """Centralized validation manager."""
    
    def __init__(self, default_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize validation manager.
        
        Args:
            default_level: Default validation level
        """
        self.default_level = default_level
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.validators: Dict[str, Callable[[Any], tuple[bool, Optional[str]]]] = {}
    
    def register_rule(
        self,
        category: str,
        rule: ValidationRule
    ):
        """
        Register validation rule.
        
        Args:
            category: Rule category
            rule: Validation rule
        """
        if category not in self.rules:
            self.rules[category] = []
        self.rules[category].append(rule)
        logger.debug(f"Registered rule {rule.name} for category {category}")
    
    def register_validator(
        self,
        name: str,
        validator: Callable[[Any], tuple[bool, Optional[str]]]
    ):
        """
        Register validator function.
        
        Args:
            name: Validator name
            validator: Validator function
        """
        self.validators[name] = validator
        logger.debug(f"Registered validator: {name}")
    
    def validate(
        self,
        category: str,
        data: Any,
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate data against rules.
        
        Args:
            category: Rule category
            data: Data to validate
            level: Validation level (uses default if not provided)
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        level = level or self.default_level
        
        if category not in self.rules:
            logger.warning(f"No rules found for category: {category}")
            return result
        
        rules = self.rules[category]
        
        for rule in rules:
            # Skip if rule level doesn't match
            if level == ValidationLevel.STRICT and rule.level != ValidationLevel.STRICT:
                continue
            if level == ValidationLevel.LENIENT and rule.level == ValidationLevel.STRICT:
                continue
            
            # Skip optional rules if data is None
            if not rule.required and data is None:
                continue
            
            # Execute validator
            try:
                is_valid, error = rule.validator(data)
                if not is_valid:
                    if error:
                        result.add_error(f"{rule.name}: {error}")
                    else:
                        result.add_error(f"{rule.name}: Validation failed")
            except Exception as e:
                result.add_error(f"{rule.name}: {str(e)}")
        
        return result
    
    def validate_with_validator(
        self,
        validator_name: str,
        data: Any
    ) -> tuple[bool, Optional[str]]:
        """
        Validate using registered validator.
        
        Args:
            validator_name: Validator name
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if validator_name not in self.validators:
            return False, f"Validator {validator_name} not found"
        
        validator = self.validators[validator_name]
        return validator(data)
    
    def get_rules(self, category: str) -> List[ValidationRule]:
        """
        Get rules for category.
        
        Args:
            category: Rule category
            
        Returns:
            List of validation rules
        """
        return self.rules.get(category, [])
    
    def clear_rules(self, category: Optional[str] = None):
        """
        Clear rules.
        
        Args:
            category: Optional category (clears all if not provided)
        """
        if category:
            self.rules.pop(category, None)
        else:
            self.rules.clear()




