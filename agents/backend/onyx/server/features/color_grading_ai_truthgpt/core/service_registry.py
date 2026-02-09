"""
Service Registry for Color Grading AI
======================================

Centralized service registry with dependency management.
"""

import logging
from typing import Dict, Any, Type, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Service definition."""
    name: str
    service_class: Type
    category: str
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    singleton: bool = True


class ServiceRegistry:
    """
    Centralized service registry.
    
    Features:
    - Service registration
    - Dependency resolution
    - Lazy loading
    - Service discovery
    """
    
    def __init__(self):
        """Initialize service registry."""
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        service_class: Type,
        category: str = "general",
        dependencies: Optional[List[str]] = None,
        optional: bool = False,
        singleton: bool = True
    ):
        """
        Register a service.
        
        Args:
            name: Service name
            service_class: Service class
            category: Service category
            dependencies: List of dependency service names
            optional: Whether service is optional
            singleton: Whether to create singleton instance
        """
        self._services[name] = ServiceDefinition(
            name=name,
            service_class=service_class,
            category=category,
            dependencies=dependencies or [],
            optional=optional,
            singleton=singleton
        )
        logger.debug(f"Registered service: {name} ({category})")
    
    def get(self, name: str, **kwargs) -> Any:
        """
        Get service instance.
        
        Args:
            name: Service name
            **kwargs: Additional arguments for service creation
            
        Returns:
            Service instance
        """
        if name not in self._services:
            raise ValueError(f"Service not registered: {name}")
        
        definition = self._services[name]
        
        # Return singleton if exists
        if definition.singleton and name in self._instances:
            return self._instances[name]
        
        # Resolve dependencies
        deps = {}
        for dep_name in definition.dependencies:
            try:
                deps[dep_name] = self.get(dep_name)
            except ValueError:
                if not definition.optional:
                    raise ValueError(f"Required dependency not found: {dep_name}")
        
        # Create instance
        instance = definition.service_class(**deps, **kwargs)
        
        # Store singleton
        if definition.singleton:
            self._instances[name] = instance
        
        return instance
    
    def list_services(self, category: Optional[str] = None) -> List[str]:
        """
        List registered services.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of service names
        """
        if category:
            return [
                name for name, defn in self._services.items()
                if defn.category == category
            ]
        return list(self._services.keys())
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get service dependencies."""
        if name not in self._services:
            return []
        return self._services[name].dependencies.copy()
    
    def clear(self):
        """Clear all services and instances."""
        self._services.clear()
        self._instances.clear()




