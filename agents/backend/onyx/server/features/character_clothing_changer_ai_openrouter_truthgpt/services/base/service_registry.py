"""
Service Registry
================
Centralized registry for service instances (singleton pattern)
"""

import logging
from typing import Dict, Any, Optional, TypeVar, Type, Callable
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """
    Centralized registry for service instances.
    
    Provides singleton pattern for services with thread-safe access.
    """
    
    def __init__(self):
        """Initialize service registry"""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = Lock()
    
    def register(
        self,
        name: str,
        service: Any,
        factory: Optional[Callable[[], Any]] = None
    ):
        """
        Register a service instance or factory.
        
        Args:
            name: Service name
            service: Service instance (if provided)
            factory: Factory function (if service not provided)
        """
        with self._lock:
            if service is not None:
                self._services[name] = service
                logger.debug(f"Registered service instance: {name}")
            elif factory is not None:
                self._factories[name] = factory
                logger.debug(f"Registered service factory: {name}")
            else:
                raise ValueError("Either service or factory must be provided")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get service instance.
        
        Args:
            name: Service name
        
        Returns:
            Service instance or None if not found
        """
        with self._lock:
            # Check if instance exists
            if name in self._services:
                return self._services[name]
            
            # Check if factory exists
            if name in self._factories:
                # Create instance using factory
                service = self._factories[name]()
                self._services[name] = service
                logger.debug(f"Created service instance from factory: {name}")
                return service
            
            return None
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a service.
        
        Args:
            name: Service name
        
        Returns:
            True if service was unregistered
        """
        with self._lock:
            removed = False
            if name in self._services:
                del self._services[name]
                removed = True
            if name in self._factories:
                del self._factories[name]
                removed = True
            
            if removed:
                logger.debug(f"Unregistered service: {name}")
            
            return removed
    
    def clear(self):
        """Clear all registered services"""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            logger.debug("Service registry cleared")
    
    def list_services(self) -> list[str]:
        """List all registered service names"""
        with self._lock:
            all_names = set(self._services.keys()) | set(self._factories.keys())
            return sorted(all_names)


# Global service registry
_service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get global service registry"""
    return _service_registry


def register_service(name: str, service: Any = None, factory: Callable[[], Any] = None):
    """
    Register a service in the global registry.
    
    Args:
        name: Service name
        service: Service instance (optional)
        factory: Factory function (optional)
    """
    _service_registry.register(name, service, factory)


def get_service(name: str) -> Optional[Any]:
    """
    Get service from global registry.
    
    Args:
        name: Service name
    
    Returns:
        Service instance or None
    """
    return _service_registry.get(name)

