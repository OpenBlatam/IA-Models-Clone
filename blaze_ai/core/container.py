import weakref
from typing import Any, Dict, Callable

class ServiceContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._weak_refs: Dict[str, weakref.ref] = {}
    
    def register_service(self, name: str, service: Any):
        """Register a service instance."""
        self._services[name] = service
    
    def register_factory(self, name: str, factory: Callable):
        """Register a service factory."""
        self._factories[name] = factory
    
    def register_singleton(self, name: str, factory: Callable):
        """Register a singleton factory."""
        self._factories[name] = factory
        self._singletons[name] = None
    
    def get_service(self, name: str) -> Any:
        """Get a service by name."""
        # Check existing instances
        if name in self._services:
            return self._services[name]
        
        # Check singletons
        if name in self._singletons:
            if self._singletons[name] is None:
                self._singletons[name] = self._factories[name]()
            return self._singletons[name]
        
        # Check factories
        if name in self._factories:
            return self._factories[name]()
        
        raise KeyError(f"Service '{name}' not found")
    
    def has_service(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._services or name in self._factories
    
    def clear(self):
        """Clear all services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._weak_refs.clear()
