"""
Advanced validation system for KV cache.

This module provides comprehensive validation capabilities including
data validation, schema validation, and integrity checks.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum


class ValidationLevel(Enum):
    """Validation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """A validation rule."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.BASIC


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)


class AdvancedCacheValidator:
    """Advanced cache validator."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self._global_rules: List[ValidationRule] = []
        self._lock = threading.Lock()
        
    def register_rule(
        self,
        name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        level: ValidationLevel = ValidationLevel.BASIC,
        key_pattern: Optional[str] = None
    ) -> None:
        """Register a validation rule."""
        rule = ValidationRule(
            name=name,
            validator=validator,
            error_message=error_message,
            level=level
        )
        
        with self._lock:
            if key_pattern:
                self._rules[key_pattern].append(rule)
            else:
                self._global_rules.append(rule)
                
    def validate(self, key: str, value: Any) -> ValidationResult:
        """Validate a key-value pair."""
        errors = []
        warnings = []
        
        with self._lock:
            # Get applicable rules
            rules_to_check = list(self._global_rules)
            
            # Check pattern-based rules
            for pattern, rules in self._rules.items():
                if self._match_pattern(key, pattern):
                    rules_to_check.extend(rules)
                    
        # Run validators
        for rule in rules_to_check:
            try:
                if not rule.validator(value):
                    if rule.level == ValidationLevel.STRICT:
                        errors.append(f"{rule.name}: {rule.error_message}")
                    else:
                        warnings.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.name}: Validation error - {str(e)}")
                
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Match key against pattern."""
        import re
        try:
            return bool(re.match(pattern, key))
        except Exception:
            return pattern == key
            
    def validate_all(self) -> Dict[str, ValidationResult]:
        """Validate all cache entries."""
        results = {}
        
        if hasattr(self.cache, '_cache'):
            for key, value in self.cache._cache.items():
                results[key] = self.validate(key, value)
                
        return results
        
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        all_results = self.validate_all()
        
        total = len(all_results)
        valid = sum(1 for r in all_results.values() if r.valid)
        invalid = total - valid
        total_errors = sum(len(r.errors) for r in all_results.values())
        total_warnings = sum(len(r.warnings) for r in all_results.values())
        
        return {
            'total_entries': total,
            'valid_entries': valid,
            'invalid_entries': invalid,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'validation_rate': valid / total if total > 0 else 0.0
        }


class ValidatedCache:
    """Cache wrapper with validation."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.validator = AdvancedCacheValidator(cache)
        self._strict_mode = False
        
    def enable_strict_mode(self) -> None:
        """Enable strict validation mode."""
        self._strict_mode = True
        
    def disable_strict_mode(self) -> None:
        """Disable strict validation mode."""
        self._strict_mode = False
        
    def get(self, key: str) -> Any:
        """Get value."""
        return self.cache.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value with validation."""
        result = self.validator.validate(key, value)
        
        if not result.valid and self._strict_mode:
            raise ValueError(f"Validation failed for key '{key}': {', '.join(result.errors)}")
        elif result.warnings:
            print(f"Validation warnings for key '{key}': {', '.join(result.warnings)}")
            
        if result.valid or not self._strict_mode:
            return self.cache.put(key, value)
        return False
        
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.cache.delete(key)
        
    def register_validation_rule(
        self,
        name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        level: ValidationLevel = ValidationLevel.BASIC,
        key_pattern: Optional[str] = None
    ) -> None:
        """Register a validation rule."""
        self.validator.register_rule(name, validator, error_message, level, key_pattern)



