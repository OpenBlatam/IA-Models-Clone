"""
Decorator Registry for centralized decorator management.

Provides a registry pattern for decorators to enable composition
and better organization.
"""

import functools
import logging
from typing import Callable, Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DecoratorType(Enum):
    """Types of decorators available."""
    LOGGING = "logging"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    CACHE = "cache"
    RETRY = "retry"
    PERFORMANCE = "performance"
    AUTH = "auth"
    ERROR_HANDLING = "error_handling"


class DecoratorRegistry:
    """Registry for managing decorators."""
    
    def __init__(self):
        self._decorators: Dict[DecoratorType, List[Callable]] = {}
    
    def register(
        self,
        decorator_type: DecoratorType,
        decorator: Callable
    ) -> None:
        """
        Register a decorator.
        
        Args:
            decorator_type: Type of decorator
            decorator: Decorator function
        """
        if decorator_type not in self._decorators:
            self._decorators[decorator_type] = []
        self._decorators[decorator_type].append(decorator)
        logger.debug(f"Registered decorator: {decorator_type.value}")
    
    def get_decorators(
        self,
        decorator_type: DecoratorType
    ) -> List[Callable]:
        """
        Get decorators of a specific type.
        
        Args:
            decorator_type: Type of decorator
        
        Returns:
            List of decorators
        """
        return self._decorators.get(decorator_type, [])
    
    def apply_decorators(
        self,
        func: Callable,
        decorator_types: List[DecoratorType]
    ) -> Callable:
        """
        Apply multiple decorators to a function.
        
        Args:
            func: Function to decorate
            decorator_types: List of decorator types to apply
        
        Returns:
            Decorated function
        """
        result = func
        for decorator_type in decorator_types:
            decorators = self.get_decorators(decorator_type)
            for decorator in decorators:
                result = decorator(result)
        return result


# Global registry
_registry = DecoratorRegistry()


def get_decorator_registry() -> DecoratorRegistry:
    """Get the global decorator registry."""
    return _registry


def compose_decorators(*decorators: Callable) -> Callable:
    """
    Compose multiple decorators into a single decorator.
    
    Args:
        *decorators: Decorators to compose
    
    Returns:
        Composed decorator
    """
    def decorator(func: Callable) -> Callable:
        result = func
        for dec in reversed(decorators):
            result = dec(result)
        return result
    return decorator

