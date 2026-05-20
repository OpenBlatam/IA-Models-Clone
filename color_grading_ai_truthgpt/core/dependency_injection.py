"""
Dependency Injection System for Color Grading AI
=================================================

Advanced dependency injection with automatic wiring and lifecycle management.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect

logger = logging.getLogger(__name__)


class ServiceScope(Enum):
    """Service scope types."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Service descriptor for DI container."""
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DependencyInjector:
    """
    Dependency injection container.
    
    Features:
    - Automatic dependency resolution
    - Service registration
    - Lifecycle management
    - Scope management
    - Circular dependency detection
    """
    
    def __init__(self):
        """Initialize dependency injector."""
        self._services: Dict[str, ServiceDescriptor] = {}
        self._instances: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        self._resolving: Set[str] = set()
    
    def register(
        self,
        service_name: str,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        scope: ServiceScope = ServiceScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        **metadata
    ):
        """
        Register service.
        
        Args:
            service_name: Service name/identifier
            service_type: Service type/interface
            implementation: Implementation class
            factory: Factory function
            scope: Service scope
            dependencies: List of dependency names
            tags: Optional tags
            **metadata: Additional metadata
        """
        if service_name in self._services:
            logger.warning(f"Service {service_name} already registered, overwriting")
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            factory=factory,
            scope=scope,
            dependencies=dependencies or [],
            tags=tags or set(),
            metadata=metadata
        )
        
        self._services[service_name] = descriptor
        logger.debug(f"Registered service: {service_name} ({scope.value})")
    
    def register_instance(self, service_name: str, instance: Any, tags: Optional[Set[str]] = None):
        """
        Register service instance.
        
        Args:
            service_name: Service name
            instance: Service instance
            tags: Optional tags
        """
        # Create descriptor for instance
        descriptor = ServiceDescriptor(
            service_type=type(instance),
            implementation=type(instance),
            scope=ServiceScope.SINGLETON,
            tags=tags or set()
        )
        
        self._services[service_name] = descriptor
        self._instances[service_name] = instance
        logger.debug(f"Registered instance: {service_name}")
    
    def resolve(
        self,
        service_name: str,
        scope_id: Optional[str] = None
    ) -> Any:
        """
        Resolve service instance.
        
        Args:
            service_name: Service name
            scope_id: Optional scope ID for scoped services
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not found or circular dependency
        """
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        
        descriptor = self._services[service_name]
        
        # Check for circular dependency
        if service_name in self._resolving:
            raise ValueError(f"Circular dependency detected: {service_name}")
        
        # Handle different scopes
        if descriptor.scope == ServiceScope.SINGLETON:
            if service_name in self._instances:
                return self._instances[service_name]
            
            instance = self._create_instance(descriptor, scope_id)
            self._instances[service_name] = instance
            return instance
        
        elif descriptor.scope == ServiceScope.SCOPED:
            if scope_id is None:
                raise ValueError(f"Scope ID required for scoped service: {service_name}")
            
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}
            
            if service_name in self._scoped_instances[scope_id]:
                return self._scoped_instances[scope_id][service_name]
            
            instance = self._create_instance(descriptor, scope_id)
            self._scoped_instances[scope_id][service_name] = instance
            return instance
        
        else:  # TRANSIENT
            return self._create_instance(descriptor, scope_id)
    
    def _create_instance(self, descriptor: ServiceDescriptor, scope_id: Optional[str]) -> Any:
        """Create service instance."""
        self._resolving.add(descriptor.service_type.__name__)
        
        try:
            # Resolve dependencies
            dependencies = {}
            for dep_name in descriptor.dependencies:
                dep_instance = self.resolve(dep_name, scope_id)
                dependencies[dep_name] = dep_instance
            
            # Use factory if provided
            if descriptor.factory:
                if dependencies:
                    return descriptor.factory(**dependencies)
                return descriptor.factory()
            
            # Auto-inject dependencies via constructor
            implementation = descriptor.implementation
            sig = inspect.signature(implementation.__init__)
            params = sig.parameters
            
            # Build kwargs for constructor
            kwargs = {}
            for param_name, param in params.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve from dependencies dict first
                if param_name in dependencies:
                    kwargs[param_name] = dependencies[param_name]
                elif dep_name := self._find_dependency_by_type(param.annotation):
                    kwargs[param_name] = self.resolve(dep_name, scope_id)
            
            # Create instance
            instance = implementation(**kwargs)
            
            return instance
        
        finally:
            self._resolving.discard(descriptor.service_type.__name__)
    
    def _find_dependency_by_type(self, annotation: Type) -> Optional[str]:
        """Find service by type annotation."""
        if annotation == inspect.Parameter.empty:
            return None
        
        for name, descriptor in self._services.items():
            if descriptor.service_type == annotation or descriptor.implementation == annotation:
                return name
        
        return None
    
    def resolve_all(self, tags: Optional[Set[str]] = None) -> List[Any]:
        """
        Resolve all services matching tags.
        
        Args:
            tags: Optional tags to filter
            
        Returns:
            List of service instances
        """
        instances = []
        
        for name in self._services.keys():
            descriptor = self._services[name]
            
            if tags:
                if not tags.intersection(descriptor.tags):
                    continue
            
            try:
                instance = self.resolve(name)
                instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to resolve {name}: {e}")
        
        return instances
    
    def get_registered_services(self) -> List[str]:
        """Get list of registered service names."""
        return list(self._services.keys())
    
    def clear_scope(self, scope_id: str):
        """Clear scoped instances for scope ID."""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
            logger.debug(f"Cleared scope: {scope_id}")
    
    def clear_all(self):
        """Clear all instances (except registered instances)."""
        self._instances.clear()
        self._scoped_instances.clear()
        logger.info("Cleared all service instances")


class ServiceRegistry:
    """
    Service registry with automatic discovery.
    
    Features:
    - Service registration
    - Automatic discovery
    - Tag-based lookup
    - Health checking
    """
    
    def __init__(self, injector: DependencyInjector):
        """
        Initialize service registry.
        
        Args:
            injector: Dependency injector
        """
        self.injector = injector
        self._service_info: Dict[str, Dict[str, Any]] = {}
    
    def register_service(
        self,
        name: str,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        scope: ServiceScope = ServiceScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        description: Optional[str] = None,
        **metadata
    ):
        """
        Register service with metadata.
        
        Args:
            name: Service name
            service_type: Service type
            implementation: Implementation class
            factory: Factory function
            scope: Service scope
            dependencies: Dependencies
            tags: Tags
            description: Service description
            **metadata: Additional metadata
        """
        # Register in injector
        self.injector.register(
            name,
            service_type,
            implementation,
            factory,
            scope,
            dependencies,
            tags,
            **metadata
        )
        
        # Store metadata
        self._service_info[name] = {
            "name": name,
            "type": service_type.__name__,
            "description": description,
            "tags": tags or set(),
            "dependencies": dependencies or [],
            "scope": scope.value,
            **metadata
        }
    
    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service information."""
        return self._service_info.get(name)
    
    def find_services_by_tag(self, tag: str) -> List[str]:
        """Find services by tag."""
        return [
            name for name, info in self._service_info.items()
            if tag in info.get("tags", set())
        ]
    
    def get_all_services(self) -> List[Dict[str, Any]]:
        """Get all registered services."""
        return list(self._service_info.values())




