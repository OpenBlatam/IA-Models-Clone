"""
Service Layer - Ultra Modular Business Logic
High-level services that orchestrate lower layers
"""

from typing import Optional, Dict, Any, List, Callable
import logging
from abc import ABC, abstractmethod

from .interfaces import IService, BaseService

logger = logging.getLogger(__name__)


# ============================================================================
# Service Configuration
# ============================================================================

class ServiceConfig:
    """Service configuration container"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """Update config"""
        self.config.update(kwargs)


# ============================================================================
# Service Registry
# ============================================================================

class ServiceRegistry:
    """Registry for service classes"""
    
    _services: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, service_class: type):
        """Register service class"""
        cls._services[name] = service_class
        logger.info(f"Registered service: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get service class"""
        return cls._services.get(name)
    
    @classmethod
    def list_services(cls) -> List[str]:
        """List all registered services"""
        return list(cls._services.keys())


# ============================================================================
# Service Container - Dependency Injection
# ============================================================================

class ServiceContainer:
    """Dependency injection container for services"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        service_class: type,
        singleton: bool = True,
        **kwargs
    ):
        """Register service"""
        if singleton:
            # Create instance immediately
            instance = service_class(**kwargs)
            self._singletons[name] = instance
        else:
            # Store factory
            self._services[name] = (service_class, kwargs)
    
    def get(self, name: str) -> Any:
        """Get service instance"""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Create from factory
        if name in self._services:
            service_class, kwargs = self._services[name]
            return service_class(**kwargs)
        
        raise ValueError(f"Service '{name}' not found")
    
    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._singletons or name in self._services
    
    def clear(self):
        """Clear all services"""
        self._services.clear()
        self._singletons.clear()


# ============================================================================
# Service Factory
# ============================================================================

class ServiceFactory:
    """Factory for creating services"""
    
    def __init__(self, container: Optional[ServiceContainer] = None):
        self.container = container or ServiceContainer()
    
    def create(
        self,
        service_type: str,
        config: Optional[ServiceConfig] = None,
        **kwargs
    ) -> IService:
        """Create service instance"""
        # Check container first
        if self.container.has(service_type):
            return self.container.get(service_type)
        
        # Get from registry
        service_class = ServiceRegistry.get(service_type)
        if not service_class:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Create with config
        service_config = config.config if config else {}
        return service_class(**{**service_config, **kwargs})


# Export main components
__all__ = [
    "ServiceConfig",
    "ServiceRegistry",
    "ServiceContainer",
    "ServiceFactory",
]



