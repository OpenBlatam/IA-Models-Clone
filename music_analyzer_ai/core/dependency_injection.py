"""
Dependency Injection Container
Manages dependencies and provides dependency injection
"""

from typing import Dict, Any, Optional, Type, Callable, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """
    Simple Dependency Injection Container
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_flags: Dict[str, bool] = {}
    
    def register(
        self,
        service_name: str,
        service_class: Type[T],
        singleton: bool = True,
        factory: Optional[Callable] = None
    ) -> None:
        """
        Register a service
        
        Args:
            service_name: Name of the service
            service_class: Class to instantiate
            singleton: Whether to use singleton pattern
            factory: Optional factory function
        """
        if factory:
            self._factories[service_name] = factory
        else:
            self._services[service_name] = service_class
        
        self._singleton_flags[service_name] = singleton
        logger.info(f"Registered service: {service_name} (singleton={singleton})")
    
    def register_instance(self, service_name: str, instance: Any) -> None:
        """Register an existing instance"""
        self._singletons[service_name] = instance
        self._singleton_flags[service_name] = True
        logger.info(f"Registered instance: {service_name}")
    
    def get(self, service_name: str, **kwargs) -> Any:
        """
        Get service instance
        
        Args:
            service_name: Name of the service
            **kwargs: Additional arguments for instantiation
        
        Returns:
            Service instance
        """
        # Check if singleton exists
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if factory exists
        if service_name in self._factories:
            instance = self._factories[service_name](**kwargs)
        elif service_name in self._services:
            service_class = self._services[service_name]
            instance = service_class(**kwargs)
        else:
            raise ValueError(f"Service '{service_name}' not registered")
        
        # Store as singleton if needed
        if self._singleton_flags.get(service_name, True):
            self._singletons[service_name] = instance
        
        return instance
    
    def has(self, service_name: str) -> bool:
        """Check if service is registered"""
        return service_name in self._services or service_name in self._factories or service_name in self._singletons
    
    def clear(self):
        """Clear all registrations"""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_flags.clear()
        logger.info("DIContainer cleared")


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get global DI container"""
    return _container


def register_service(
    service_name: str,
    service_class: Type[T],
    singleton: bool = True,
    factory: Optional[Callable] = None
) -> None:
    """Register a service in global container"""
    _container.register(service_name, service_class, singleton, factory)


def get_service(service_name: str, **kwargs) -> Any:
    """Get service from global container"""
    return _container.get(service_name, **kwargs)









