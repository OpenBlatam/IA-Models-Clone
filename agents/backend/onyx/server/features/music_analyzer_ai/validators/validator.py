"""
Validator Interfaces and Base Classes
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class IValidator(ABC):
    """
    Interface for validators
    """
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data"""
        pass


class BaseValidator(IValidator):
    """
    Base validator implementation
    """
    
    def __init__(self, name: str = "BaseValidator"):
        self.name = name
    
    def validate(self, data: Any) -> ValidationResult:
        """Base validation - override in subclasses"""
        return ValidationResult(is_valid=True, errors=[], warnings=[])
    
    def _add_error(self, result: ValidationResult, error: str) -> None:
        """Add error to result"""
        result.is_valid = False
        result.errors.append(error)
    
    def _add_warning(self, result: ValidationResult, warning: str) -> None:
        """Add warning to result"""
        result.warnings.append(warning)


class CompositeValidator(BaseValidator):
    """
    Validator that combines multiple validators
    """
    
    def __init__(self, validators: List[IValidator], name: str = "CompositeValidator"):
        super().__init__(name)
        self.validators = validators
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate using all validators"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for validator in self.validators:
            validator_result = validator.validate(data)
            result.errors.extend(validator_result.errors)
            result.warnings.extend(validator_result.warnings)
            if not validator_result.is_valid:
                result.is_valid = False
        
        return result








