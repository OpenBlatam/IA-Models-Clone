"""
Service Factory for creating and managing service instances.

Provides a centralized way to create services with proper dependency injection
and lifecycle management.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from functools import lru_cache

from .base_service import BaseService
from .service_registry import get_service_registry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseService)


class ServiceFactory:
    """Factory for creating service instances."""
    
    def __init__(self):
        self._service_classes: Dict[str, Type[BaseService]] = {}
        self._service_instances: Dict[str, BaseService] = {}
        self._factories: Dict[str, Callable[[], BaseService]] = {}
    
    def register(
        self, 
        name: str, 
        service_class: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = True
    ) -> None:
        """
        Register a service class or factory.
        
        Args:
            name: Service name
            service_class: Service class to register
            factory: Factory function to create service instances
            singleton: Whether to use singleton pattern
        """
        if factory:
            self._factories[name] = factory
        elif service_class:
            self._service_classes[name] = service_class
        else:
            raise ValueError("Either service_class or factory must be provided")
        
        logger.debug(f"Registered service: {name} (singleton={singleton})")
    
    def create(self, name: str, **kwargs) -> Optional[BaseService]:
        """
        Create a service instance.
        
        Args:
            name: Service name
            **kwargs: Additional arguments for service initialization
        
        Returns:
            Service instance or None if not found
        """
        # Check if factory exists
        if name in self._factories:
            return self._factories[name](**kwargs)
        
        # Check if class exists
        if name in self._service_classes:
            service_class = self._service_classes[name]
            
            # Check for singleton instance
            if name in self._service_instances:
                return self._service_instances[name]
            
            # Create new instance
            instance = service_class(**kwargs)
            self._service_instances[name] = instance
            return instance
        
        logger.warning(f"Service not found: {name}")
        return None
    
    def get(self, name: str) -> Optional[BaseService]:
        """
        Get existing service instance.
        
        Args:
            name: Service name
        
        Returns:
            Service instance or None
        """
        return self._service_instances.get(name)
    
    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear service instances.
        
        Args:
            name: Service name to clear, or None to clear all
        """
        if name:
            self._service_instances.pop(name, None)
        else:
            self._service_instances.clear()


# Global factory instance
_factory = ServiceFactory()


def get_service_factory() -> ServiceFactory:
    """Get the global service factory."""
    return _factory


def register_service(
    name: str,
    service_class: Optional[Type[T]] = None,
    factory: Optional[Callable[[], T]] = None
) -> None:
    """Register a service with the global factory."""
    _factory.register(name, service_class=service_class, factory=factory)


def create_service(name: str, **kwargs) -> Optional[BaseService]:
    """Create a service instance using the global factory."""
    return _factory.create(name, **kwargs)


def get_service(name: str) -> Optional[BaseService]:
    """Get a service instance from the global factory."""
    return _factory.get(name)

