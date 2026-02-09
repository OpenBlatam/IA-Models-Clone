"""
Dependency Injection Container for Piel Mejorador AI SAM3
========================================================

Dependency injection system for better code organization.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceDefinition:
    """Service definition for DI container."""
    service_type: Type
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    singleton: bool = True


class DIContainer:
    """
    Dependency Injection Container.
    
    Features:
    - Service registration
    - Singleton and transient services
    - Factory functions
    - Service resolution
    """
    
    def __init__(self):
        """Initialize DI container."""
        self._services: Dict[Type, ServiceDefinition] = {}
        self._instances: Dict[Type, Any] = {}
    
    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable] = None,
        instance: Optional[T] = None,
        singleton: bool = True
    ):
        """
        Register a service.
        
        Args:
            service_type: Service type/interface
            factory: Optional factory function
            instance: Optional pre-created instance
            singleton: Whether service is singleton
        """
        self._services[service_type] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            instance=instance,
            singleton=singleton
        )
        logger.debug(f"Registered service: {service_type.__name__}")
    
    def register_instance(self, service_type: Type[T], instance: T):
        """
        Register a pre-created instance.
        
        Args:
            service_type: Service type
            instance: Service instance
        """
        self.register(service_type, instance=instance, singleton=True)
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]):
        """
        Register a factory function.
        
        Args:
            service_type: Service type
            factory: Factory function
        """
        self.register(service_type, factory=factory, singleton=False)
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        definition = self._services[service_type]
        
        # Return pre-created instance
        if definition.instance is not None:
            return definition.instance
        
        # Check singleton cache
        if definition.singleton and service_type in self._instances:
            return self._instances[service_type]
        
        # Create instance
        if definition.factory:
            instance = definition.factory()
        else:
            # Try to instantiate directly
            instance = service_type()
        
        # Cache singleton
        if definition.singleton:
            self._instances[service_type] = instance
        
        return instance
    
    def get(self, service_type: Type[T]) -> Optional[T]:
        """
        Get service if registered, None otherwise.
        
        Args:
            service_type: Service type
            
        Returns:
            Service instance or None
        """
        try:
            return self.resolve(service_type)
        except ValueError:
            return None
    
    def clear(self):
        """Clear all registered services."""
        self._services.clear()
        self._instances.clear()




