"""
Service Factory - Centralized service creation and management
"""

from typing import Dict, Type, Any, Optional
from services.domains import get_service, get_all_services, auto_discover_services

_service_instances: Dict[str, Any] = {}


class ServiceFactory:
    """Factory for creating and managing service instances"""
    
    def __init__(self):
        """Initialize the service factory"""
        auto_discover_services()
        self._instances: Dict[str, Any] = {}
    
    def get_service(self, domain: str, service_name: str, singleton: bool = True) -> Any:
        """
        Get a service instance
        
        Args:
            domain: Service domain (e.g., 'assessment', 'recovery')
            service_name: Name of the service
            singleton: Whether to return a singleton instance
        
        Returns:
            Service instance
        """
        key = f"{domain}.{service_name}"
        
        if singleton and key in self._instances:
            return self._instances[key]
        
        try:
            service = get_service(domain, service_name)
            if singleton:
                self._instances[key] = service
            return service
        except ValueError as e:
            raise ValueError(f"Service {key} not found: {e}")
    
    def register_service_instance(self, domain: str, service_name: str, instance: Any) -> None:
        """Register a service instance"""
        key = f"{domain}.{service_name}"
        self._instances[key] = instance
    
    def list_available_services(self) -> Dict[str, Type[Any]]:
        """List all available services"""
        return get_all_services()
    
    def clear_cache(self) -> None:
        """Clear all cached service instances"""
        self._instances.clear()


_global_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get the global service factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = ServiceFactory()
    return _global_factory


def get_service_instance(domain: str, service_name: str, singleton: bool = True) -> Any:
    """Convenience function to get a service instance"""
    return get_service_factory().get_service(domain, service_name, singleton)



