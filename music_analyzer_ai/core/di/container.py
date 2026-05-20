"""
Dependency Injection Container Module

Manages dependencies and provides dependency injection with automatic dependency resolution.
"""

from typing import Dict, Any, Optional, Type, Callable, TypeVar, List
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """
    Enhanced Dependency Injection Container with automatic dependency resolution.
    """
    
    def __init__(self):
        self._services: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_flags: Dict[str, bool] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._scoped: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        service_name: str,
        service_class: Type[T],
        singleton: bool = True,
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a service with optional dependency specification.
        
        Args:
            service_name: Name of the service.
            service_class: Class to instantiate.
            singleton: Whether to use singleton pattern.
            factory: Optional factory function.
            dependencies: Optional list of dependency service names.
        """
        if factory:
            self._factories[service_name] = factory
        else:
            self._services[service_name] = service_class
        
        self._singleton_flags[service_name] = singleton
        
        # Store explicit dependencies or auto-detect from constructor
        if dependencies:
            self._dependencies[service_name] = dependencies
        else:
            # Try to auto-detect dependencies from constructor
            self._dependencies[service_name] = self._detect_dependencies(service_class)
        
        logger.debug(f"Registered service: {service_name} (singleton={singleton}, deps={self._dependencies[service_name]})")
    
    def _detect_dependencies(self, service_class: Type) -> List[str]:
        """
        Auto-detect dependencies from constructor signature.
        
        Args:
            service_class: Class to inspect.
        
        Returns:
            List of detected dependency names.
        """
        try:
            sig = inspect.signature(service_class.__init__)
            deps = []
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param_name != 'args' and param_name != 'kwargs':
                    # Try to infer service name from parameter name
                    # Convert snake_case to service name (e.g., spotify_service -> spotify_service)
                    deps.append(param_name)
            return deps
        except Exception as e:
            logger.warning(f"Could not detect dependencies for {service_class}: {e}")
            return []
    
    def register_instance(self, service_name: str, instance: Any) -> None:
        """
        Register an existing instance.
        
        Args:
            service_name: Name of the service.
            instance: Service instance.
        """
        self._singletons[service_name] = instance
        self._singleton_flags[service_name] = True
        logger.debug(f"Registered instance: {service_name}")
    
    def get(self, service_name: str, scope: Optional[str] = None, **kwargs) -> Any:
        """
        Get service instance with automatic dependency resolution.
        
        Args:
            service_name: Name of the service.
            scope: Optional scope identifier for scoped instances.
            **kwargs: Additional arguments for instantiation (overrides resolved deps).
        
        Returns:
            Service instance.
        """
        # Check scoped instances
        if scope and scope in self._scoped:
            if service_name in self._scoped[scope]:
                return self._scoped[scope][service_name]
        
        # Check if singleton exists
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Resolve dependencies
        resolved_deps = self._resolve_dependencies(service_name, scope)
        resolved_deps.update(kwargs)  # Override with explicit kwargs
        
        # Create instance
        if service_name in self._factories:
            instance = self._factories[service_name](**resolved_deps)
        elif service_name in self._services:
            service_class = self._services[service_name]
            instance = service_class(**resolved_deps)
        else:
            raise ValueError(f"Service '{service_name}' not registered")
        
        # Store instance
        if scope:
            if scope not in self._scoped:
                self._scoped[scope] = {}
            self._scoped[scope][service_name] = instance
        elif self._singleton_flags.get(service_name, True):
            self._singletons[service_name] = instance
        
        return instance
    
    def _resolve_dependencies(self, service_name: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve dependencies for a service.
        
        Args:
            service_name: Name of the service.
            scope: Optional scope identifier.
        
        Returns:
            Dictionary of resolved dependencies.
        """
        deps = {}
        if service_name in self._dependencies:
            for dep_name in self._dependencies[service_name]:
                try:
                    deps[dep_name] = self.get(dep_name, scope=scope)
                except ValueError as e:
                    logger.warning(f"Could not resolve dependency '{dep_name}' for '{service_name}': {e}")
        return deps
    
    def has(self, service_name: str) -> bool:
        """
        Check if service is registered.
        
        Args:
            service_name: Name of the service.
        
        Returns:
            True if service is registered.
        """
        return (
            service_name in self._services or
            service_name in self._factories or
            service_name in self._singletons
        )
    
    def clear_scope(self, scope: str) -> None:
        """Clear all instances in a scope."""
        if scope in self._scoped:
            del self._scoped[scope]
            logger.debug(f"Cleared scope: {scope}")
    
    def clear(self):
        """Clear all registrations."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_flags.clear()
        self._dependencies.clear()
        self._scoped.clear()
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
    factory: Optional[Callable] = None,
    dependencies: Optional[List[str]] = None
) -> None:
    """Register a service in global container"""
    _container.register(service_name, service_class, singleton, factory, dependencies)


def get_service(service_name: str, scope: Optional[str] = None, **kwargs) -> Any:
    """Get service from global container"""
    return _container.get(service_name, scope=scope, **kwargs)



