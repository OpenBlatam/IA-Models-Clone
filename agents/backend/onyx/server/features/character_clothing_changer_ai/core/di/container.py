"""
Dependency Injection Container
==============================

Simple dependency injection container for service management.
"""

from typing import Dict, Any, Optional, Type, TypeVar, Callable
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """Dependency injection container."""
    
    def __init__(self):
        """Initialize DI container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        service: Any,
        singleton: bool = True,
    ) -> None:
        """
        Register a service.
        
        Args:
            name: Service name
            service: Service instance or class
            singleton: Whether to treat as singleton
        """
        if singleton:
            self._singletons[name] = service
        else:
            self._services[name] = service
        
        logger.debug(f"Registered service: {name} (singleton: {singleton})")
    
    def register_factory(
        self,
        name: str,
        factory: Callable,
    ) -> None:
        """
        Register a factory function.
        
        Args:
            name: Service name
            factory: Factory function
        """
        self._factories[name] = factory
        logger.debug(f"Registered factory: {name}")
    
    def get(
        self,
        name: str,
        default: Optional[Any] = None,
    ) -> Any:
        """
        Get a service.
        
        Args:
            name: Service name
            default: Default value if not found
            
        Returns:
            Service instance
        """
        # Check singletons
        if name in self._singletons:
            return self._singletons[name]
        
        # Check services
        if name in self._services:
            return self._services[name]
        
        # Check factories
        if name in self._factories:
            service = self._factories[name]()
            # Cache as singleton if not already cached
            if name not in self._singletons:
                self._singletons[name] = service
            return service
        
        if default is not None:
            return default
        
        raise ValueError(f"Service not found: {name}")
    
    def has(self, name: str) -> bool:
        """
        Check if service is registered.
        
        Args:
            name: Service name
            
        Returns:
            True if registered
        """
        return (
            name in self._services or
            name in self._singletons or
            name in self._factories
        )
    
    def remove(self, name: str) -> None:
        """
        Remove a service.
        
        Args:
            name: Service name
        """
        self._services.pop(name, None)
        self._singletons.pop(name, None)
        self._factories.pop(name, None)
        logger.debug(f"Removed service: {name}")
    
    def clear(self) -> None:
        """Clear all services."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()
        logger.info("DI container cleared")
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered service names.
        
        Returns:
            Dictionary of service names and types
        """
        services = {}
        
        for name in self._services:
            services[name] = type(self._services[name]).__name__
        
        for name in self._singletons:
            services[name] = type(self._singletons[name]).__name__
        
        for name in self._factories:
            services[name] = "factory"
        
        return services


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get global DI container."""
    return _container


def register_service(
    name: str,
    service: Any,
    singleton: bool = True,
) -> None:
    """Register a service in global container."""
    _container.register(name, service, singleton)


def get_service(name: str, default: Optional[Any] = None) -> Any:
    """Get a service from global container."""
    return _container.get(name, default)

