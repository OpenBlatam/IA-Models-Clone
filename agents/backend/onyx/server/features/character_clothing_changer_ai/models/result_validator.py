"""
Result Validator for Flux2 Clothing Changer
===========================================

Advanced result validation and quality checking.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    score: float
    level: ValidationLevel
    checks_passed: int
    checks_failed: int
    details: Dict[str, Any] = None
    timestamp: float = time.time()
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ResultValidator:
    """Advanced result validation system."""
    
    def __init__(
        self,
        default_level: ValidationLevel = ValidationLevel.STANDARD,
    ):
        """
        Initialize result validator.
        
        Args:
            default_level: Default validation level
        """
        self.default_level = default_level
        self.validators: Dict[str, callable] = {}
        self.validation_history: List[ValidationResult] = []
    
    def register_validator(
        self,
        validator_name: str,
        validator_func: callable,
    ) -> None:
        """
        Register validator function.
        
        Args:
            validator_name: Validator name
            validator_func: Validator function
        """
        self.validators[validator_name] = validator_func
        logger.info(f"Registered validator: {validator_name}")
    
    def validate(
        self,
        result: Any,
        level: Optional[ValidationLevel] = None,
        validators: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate result.
        
        Args:
            result: Result to validate
            level: Optional validation level
            validators: Optional list of validator names
            
        Returns:
            Validation result
        """
        level = level or self.default_level
        
        if validators is None:
            validators = list(self.validators.keys())
        
        checks_passed = 0
        checks_failed = 0
        details = {}
        
        for validator_name in validators:
            if validator_name not in self.validators:
                continue
            
            validator_func = self.validators[validator_name]
            
            try:
                is_valid, detail = validator_func(result)
                if is_valid:
                    checks_passed += 1
                else:
                    checks_failed += 1
                details[validator_name] = {
                    "passed": is_valid,
                    "detail": detail,
                }
            except Exception as e:
                checks_failed += 1
                details[validator_name] = {
                    "passed": False,
                    "error": str(e),
                }
        
        total_checks = checks_passed + checks_failed
        score = checks_passed / total_checks if total_checks > 0 else 0.0
        
        # Determine if overall valid based on level
        if level == ValidationLevel.STRICT:
            is_valid = checks_failed == 0
        elif level == ValidationLevel.STANDARD:
            is_valid = score >= 0.8
        else:  # BASIC
            is_valid = score >= 0.5
        
        validation_result = ValidationResult(
            is_valid=is_valid,
            score=score,
            level=level,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details=details,
        )
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        if not self.validation_history:
            return {}
        
        total_validations = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v.is_valid)
        avg_score = sum(v.score for v in self.validation_history) / total_validations
        
        return {
            "total_validations": total_validations,
            "valid_results": valid_count,
            "invalid_results": total_validations - valid_count,
            "average_score": avg_score,
            "validators_registered": len(self.validators),
        }


