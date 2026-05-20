"""
Service Locator - Centralized service instances for dependency injection
This allows routers to access services without circular dependencies
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ServiceLocator:
    """Centralized service locator for dependency injection"""
    
    def __init__(self):
        self._services = {}
        self._initialized = False
    
    def register_service(self, name: str, service):
        """Register a service instance"""
        self._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get_service(self, name: str):
        """Get a service instance"""
        if name not in self._services:
            raise ValueError(f"Service {name} not registered")
        return self._services[name]
    
    def initialize_services(self, services_dict: dict):
        """Initialize all services from a dictionary"""
        for name, service in services_dict.items():
            self.register_service(name, service)
        self._initialized = True
        logger.info(f"Initialized {len(services_dict)} services")
    
    def is_initialized(self) -> bool:
        """Check if services are initialized"""
        return self._initialized


# Global service locator instance
_service_locator: Optional[ServiceLocator] = None


def get_service_locator() -> ServiceLocator:
    """Get or create the global service locator"""
    global _service_locator
    if _service_locator is None:
        _service_locator = ServiceLocator()
    return _service_locator


def get_service(name: str):
    """Convenience function to get a service"""
    return get_service_locator().get_service(name)




