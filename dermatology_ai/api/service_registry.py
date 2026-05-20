"""
Service registry for dynamic service initialization
Refactored from dermatology_api_modular.py to reduce code duplication
"""

from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Registry for service initialization"""
    
    def __init__(self):
        self._services: Dict[str, Callable] = {}
        self._service_groups: Dict[str, List[str]] = {}
    
    def register(
        self,
        service_name: str,
        factory: Callable,
        group: Optional[str] = None
    ) -> None:
        """
        Register a service factory
        
        Args:
            service_name: Name of the service
            factory: Factory function or class to create the service
            group: Optional group name for organizing services
        """
        self._services[service_name] = factory
        if group:
            if group not in self._service_groups:
                self._service_groups[group] = []
            self._service_groups[group].append(service_name)
    
    def register_group(
        self,
        group_name: str,
        services: Dict[str, Callable]
    ) -> None:
        """
        Register a group of services
        
        Args:
            group_name: Name of the service group
            services: Dictionary of service names to factories
        """
        for service_name, factory in services.items():
            self.register(service_name, factory, group=group_name)
    
    def initialize_service(self, service_name: str, *args, **kwargs) -> Any:
        """
        Initialize a service by name
        
        Args:
            service_name: Name of the service to initialize
            *args: Positional arguments for the factory
            **kwargs: Keyword arguments for the factory
        
        Returns:
            Initialized service instance
        
        Raises:
            KeyError: If service is not registered
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' is not registered")
        
        factory = self._services[service_name]
        return factory(*args, **kwargs)
    
    def initialize_all(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Initialize all registered services
        
        Args:
            *args: Positional arguments for factories
            **kwargs: Keyword arguments for factories
        
        Returns:
            Dictionary of service names to initialized instances
        """
        initialized = {}
        for service_name, factory in self._services.items():
            try:
                initialized[service_name] = factory(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize service '{service_name}': {e}")
        return initialized
    
    def initialize_group(self, group_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Initialize all services in a group
        
        Args:
            group_name: Name of the group to initialize
            *args: Positional arguments for factories
            **kwargs: Keyword arguments for factories
        
        Returns:
            Dictionary of service names to initialized instances
        """
        if group_name not in self._service_groups:
            return {}
        
        initialized = {}
        for service_name in self._service_groups[group_name]:
            try:
                factory = self._services[service_name]
                initialized[service_name] = factory(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize service '{service_name}': {e}")
        return initialized
    
    def get_registered_services(self) -> List[str]:
        """Get list of all registered service names"""
        return list(self._services.keys())
    
    def get_service_groups(self) -> Dict[str, List[str]]:
        """Get all service groups"""
        return self._service_groups.copy()


_global_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get or create the global service registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry







