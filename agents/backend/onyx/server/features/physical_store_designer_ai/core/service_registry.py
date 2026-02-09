"""
Service registry for managing service instances across routes
"""

from typing import Dict, Any, Type, TypeVar
from .factories import ServiceFactory
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """Registry for managing service instances"""
    
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, service_class: Type[T], use_factory: bool = False) -> T:
        """Register a service instance"""
        if name not in cls._services:
            if use_factory:
                cls._services[name] = ServiceFactory.get_service(name, service_class)
            else:
                cls._services[name] = service_class()
            logger.debug(f"Registered service: {name}")
        return cls._services[name]
    
    @classmethod
    def get(cls, name: str) -> Any:
        """Get a registered service"""
        if name not in cls._services:
            raise ValueError(f"Service '{name}' not registered")
        return cls._services[name]
    
    @classmethod
    def get_or_register(cls, name: str, service_class: Type[T], use_factory: bool = False) -> T:
        """Get service if exists, otherwise register it"""
        if name not in cls._services:
            return cls.register(name, service_class, use_factory)
        return cls._services[name]
    
    @classmethod
    def clear(cls):
        """Clear all registered services"""
        cls._services.clear()
        logger.debug("Service registry cleared")
    
    @classmethod
    def list_services(cls) -> list[str]:
        """List all registered service names"""
        return list(cls._services.keys())

