"""
Dependency Injection Container
==============================

Simple dependency injection container.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DependencyContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        """Initialize container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(
        self,
        service_type: Union[Type[T], str],
        instance: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = True,
        alias: Optional[str] = None
    ):
        """
        Register a service.
        
        Args:
            service_type: Service type or name
            instance: Service instance (if provided)
            factory: Factory function (if provided)
            singleton: Whether to treat as singleton
            alias: Optional alias name
        """
        name = service_type if isinstance(service_type, str) else service_type.__name__
        
        if instance is not None:
            self._services[name] = instance
            if singleton:
                self._singletons[name] = instance
        elif factory is not None:
            self._factories[name] = factory
            if singleton:
                # Create instance immediately for singleton
                self._singletons[name] = factory()
        else:
            raise ValueError("Either instance or factory must be provided")
        
        if alias:
            self._aliases[alias] = name
        
        logger.debug(f"Registered service: {name}")
    
    def get(self, service_type: Union[Type[T], str]) -> T:
        """
        Get service instance.
        
        Args:
            service_type: Service type or name
            
        Returns:
            Service instance
        """
        name = service_type if isinstance(service_type, str) else service_type.__name__
        
        # Check alias
        if name in self._aliases:
            name = self._aliases[name]
        
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Check services
        if name in self._services:
            return self._services[name]
        
        # Check factories
        if name in self._factories:
            instance = self._factories[name]()
            if name in self._singletons:
                self._singletons[name] = instance
            return instance
        
        raise ValueError(f"Service not found: {name}")
    
    def has(self, service_type: Union[Type[T], str]) -> bool:
        """
        Check if service is registered.
        
        Args:
            service_type: Service type or name
            
        Returns:
            True if registered
        """
        name = service_type if isinstance(service_type, str) else service_type.__name__
        
        if name in self._aliases:
            name = self._aliases[name]
        
        return (
            name in self._services or
            name in self._factories or
            name in self._singletons
        )
    
    def unregister(self, service_type: Union[Type[T], str]):
        """
        Unregister a service.
        
        Args:
            service_type: Service type or name
        """
        name = service_type if isinstance(service_type, str) else service_type.__name__
        
        self._services.pop(name, None)
        self._factories.pop(name, None)
        self._singletons.pop(name, None)
        
        # Remove aliases pointing to this service
        self._aliases = {
            alias: target
            for alias, target in self._aliases.items()
            if target != name
        }
        
        logger.debug(f"Unregistered service: {name}")
    
    def clear(self):
        """Clear all services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._aliases.clear()


# Global container instance
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get global dependency container."""
    return _container


def register(
    service_type: Union[Type[T], str],
    instance: Optional[T] = None,
    factory: Optional[Callable[[], T]] = None,
    singleton: bool = True,
    alias: Optional[str] = None
):
    """Register service in global container."""
    _container.register(service_type, instance, factory, singleton, alias)


def get(service_type: Union[Type[T], str]) -> T:
    """Get service from global container."""
    return _container.get(service_type)


def has(service_type: Union[Type[T], str]) -> bool:
    """Check if service is registered in global container."""
    return _container.has(service_type)

