"""
Factory Pattern Utilities for Piel Mejorador AI SAM3
====================================================

Unified factory pattern implementation utilities.
"""

import logging
from typing import TypeVar, Callable, Dict, Any, Optional, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class FactoryMethod:
    """Factory method definition."""
    key: str
    factory_func: Callable[..., T]
    description: Optional[str] = None


class GenericFactory:
    """Generic factory for creating objects."""
    
    def __init__(self):
        """Initialize factory."""
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._descriptions: Dict[str, str] = {}
    
    def register(
        self,
        key: str,
        factory_func: Callable[..., T],
        description: Optional[str] = None
    ):
        """
        Register factory method.
        
        Args:
            key: Factory key
            factory_func: Factory function
            description: Optional description
        """
        self._factories[key] = factory_func
        if description:
            self._descriptions[key] = description
        logger.debug(f"Registered factory: {key}")
    
    def create(self, key: str, *args, **kwargs) -> Any:
        """
        Create object using factory.
        
        Args:
            key: Factory key
            *args: Positional arguments for factory
            **kwargs: Keyword arguments for factory
            
        Returns:
            Created object
            
        Raises:
            KeyError: If factory not found
        """
        if key not in self._factories:
            raise KeyError(f"Factory not found: {key}")
        
        factory_func = self._factories[key]
        try:
            return factory_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error creating object with factory {key}: {e}")
            raise
    
    def can_create(self, key: str) -> bool:
        """
        Check if factory can create object.
        
        Args:
            key: Factory key
            
        Returns:
            True if factory exists
        """
        return key in self._factories
    
    def list_factories(self) -> Dict[str, str]:
        """
        List all registered factories.
        
        Returns:
            Dictionary of factory keys and descriptions
        """
        return {
            key: self._descriptions.get(key, "No description")
            for key in self._factories.keys()
        }
    
    def unregister(self, key: str):
        """
        Unregister factory.
        
        Args:
            key: Factory key
        """
        self._factories.pop(key, None)
        self._descriptions.pop(key, None)


class FactoryUtils:
    """Unified factory pattern utilities."""
    
    @staticmethod
    def create_factory() -> GenericFactory:
        """
        Create generic factory.
        
        Returns:
            GenericFactory instance
        """
        return GenericFactory()
    
    @staticmethod
    def create_factory_method(
        key: str,
        factory_func: Callable[..., T],
        description: Optional[str] = None
    ) -> FactoryMethod:
        """
        Create factory method definition.
        
        Args:
            key: Factory key
            factory_func: Factory function
            description: Optional description
            
        Returns:
            FactoryMethod object
        """
        return FactoryMethod(
            key=key,
            factory_func=factory_func,
            description=description
        )


# Convenience functions
def create_factory() -> GenericFactory:
    """Create factory."""
    return FactoryUtils.create_factory()


def create_factory_method(key: str, factory_func: Callable[..., T], **kwargs) -> FactoryMethod:
    """Create factory method."""
    return FactoryUtils.create_factory_method(key, factory_func, **kwargs)




