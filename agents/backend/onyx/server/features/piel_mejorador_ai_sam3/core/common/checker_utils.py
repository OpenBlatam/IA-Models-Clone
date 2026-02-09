"""
Checker Utilities for Piel Mejorador AI SAM3
============================================

Unified checker and validator pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CheckResult:
    """Result of a check."""
    valid: bool
    message: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize errors list."""
        if self.errors is None:
            self.errors = []


class Checker(ABC):
    """Base checker interface."""
    
    @abstractmethod
    def check(self, value: T) -> CheckResult:
        """Check value."""
        pass


class FunctionChecker(Checker):
    """Checker using a function."""
    
    def __init__(
        self,
        check_func: Callable[[T], bool],
        error_message: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize function checker.
        
        Args:
            check_func: Check function
            error_message: Optional error message
            name: Optional checker name
        """
        self._check_func = check_func
        self._error_message = error_message or "Check failed"
        self.name = name or check_func.__name__
    
    def check(self, value: T) -> CheckResult:
        """Check value."""
        try:
            valid = self._check_func(value)
            if valid:
                return CheckResult(valid=True)
            return CheckResult(valid=False, message=self._error_message)
        except Exception as e:
            return CheckResult(
                valid=False,
                message=f"Check error: {str(e)}",
                errors=[str(e)]
            )


class CompositeChecker(Checker):
    """Checker that combines multiple checkers."""
    
    def __init__(
        self,
        *checkers: Checker,
        require_all: bool = True
    ):
        """
        Initialize composite checker.
        
        Args:
            *checkers: Checkers to combine
            require_all: If True, all must pass; if False, any must pass
        """
        self._checkers = checkers
        self._require_all = require_all
    
    def check(self, value: T) -> CheckResult:
        """Check value."""
        results = [checker.check(value) for checker in self._checkers]
        
        if self._require_all:
            valid = all(r.valid for r in results)
            if valid:
                return CheckResult(valid=True)
            
            errors = []
            for r in results:
                if not r.valid:
                    errors.extend(r.errors or [r.message] if r.message else [])
            
            return CheckResult(
                valid=False,
                message="One or more checks failed",
                errors=errors
            )
        else:
            valid = any(r.valid for r in results)
            if valid:
                return CheckResult(valid=True)
            
            errors = []
            for r in results:
                if not r.valid:
                    errors.extend(r.errors or [r.message] if r.message else [])
            
            return CheckResult(
                valid=False,
                message="All checks failed",
                errors=errors
            )


class CheckerUtils:
    """Unified checker utilities."""
    
    @staticmethod
    def create_checker(
        check_func: Callable[[T], bool],
        error_message: Optional[str] = None,
        name: Optional[str] = None
    ) -> FunctionChecker:
        """
        Create checker from function.
        
        Args:
            check_func: Check function
            error_message: Optional error message
            name: Optional checker name
            
        Returns:
            FunctionChecker
        """
        return FunctionChecker(check_func, error_message, name)
    
    @staticmethod
    def create_composite_checker(
        *checkers: Checker,
        require_all: bool = True
    ) -> CompositeChecker:
        """
        Create composite checker.
        
        Args:
            *checkers: Checkers to combine
            require_all: If True, all must pass
            
        Returns:
            CompositeChecker
        """
        return CompositeChecker(*checkers, require_all=require_all)
    
    @staticmethod
    def not_none_checker() -> Checker:
        """Create not None checker."""
        return FunctionChecker(
            lambda x: x is not None,
            error_message="Value cannot be None",
            name="not_none"
        )
    
    @staticmethod
    def not_empty_checker() -> Checker:
        """Create not empty checker."""
        def check_empty(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (str, list, dict, tuple, set)):
                return len(value) > 0
            return True
        
        return FunctionChecker(
            check_empty,
            error_message="Value cannot be empty",
            name="not_empty"
        )
    
    @staticmethod
    def type_checker(expected_type: type) -> Checker:
        """Create type checker."""
        return FunctionChecker(
            lambda x: isinstance(x, expected_type),
            error_message=f"Value must be of type {expected_type.__name__}",
            name=f"type_{expected_type.__name__}"
        )
    
    @staticmethod
    def range_checker(min_value: Optional[float] = None, max_value: Optional[float] = None) -> Checker:
        """Create range checker."""
        def check_range(value: Any) -> bool:
            if not isinstance(value, (int, float)):
                return False
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            return True
        
        return FunctionChecker(
            check_range,
            error_message=f"Value must be in range [{min_value or '-∞'}, {max_value or '∞'}]",
            name="range"
        )
    
    @staticmethod
    def length_checker(min_length: Optional[int] = None, max_length: Optional[int] = None) -> Checker:
        """Create length checker."""
        def check_length(value: Any) -> bool:
            if not hasattr(value, '__len__'):
                return False
            length = len(value)
            if min_length is not None and length < min_length:
                return False
            if max_length is not None and length > max_length:
                return False
            return True
        
        return FunctionChecker(
            check_length,
            error_message=f"Length must be in range [{min_length or '0'}, {max_length or '∞'}]",
            name="length"
        )


# Convenience functions
def create_checker(check_func: Callable[[T], bool], **kwargs) -> FunctionChecker:
    """Create checker."""
    return CheckerUtils.create_checker(check_func, **kwargs)


def create_composite_checker(*checkers: Checker, **kwargs) -> CompositeChecker:
    """Create composite checker."""
    return CheckerUtils.create_composite_checker(*checkers, **kwargs)




