"""
Dependency Injection Container

Enhanced DI container with automatic dependency resolution, scopes, and circular dependency detection.
"""

from typing import Dict, Any, Optional, Type, TypeVar, Callable, List
import inspect

from config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

# Global container instance
_container: Optional['DIContainer'] = None


class DIContainer:
    """
    Enhanced Dependency Injection Container.
    
    Supports:
    - Automatic dependency resolution
    - Singleton, transient, and scoped patterns
    - Circular dependency detection
    - Factory functions
    """
    
    def __init__(self):
        self._services: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_flags: Dict[str, bool] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._scoped: Dict[str, Dict[str, Any]] = {}
        self._resolving: set = set()  # Track services being resolved (for circular deps)
    
    def register(
        self,
        name: str,
        service: Type[T] | Callable,
        singleton: bool = True,
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a service with improved validation and error handling.
        
        Args:
            name: Service name
            service: Service class or callable
            singleton: Whether to use singleton pattern
            factory: Optional factory function
            dependencies: Optional explicit list of dependency names
            
        Raises:
            ValueError: If service name is invalid or already registered
            TypeError: If service type is invalid
        """
        # Validaciones
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Service name must be a non-empty string")
        
        name = name.strip()
        
        if name in self._services or name in self._factories or name in self._singletons:
            logger.warning(f"Service '{name}' ya está registrado. Sobrescribiendo...")
        
        if factory and not callable(factory):
            raise TypeError(f"Factory for service '{name}' must be callable")
        
        if not factory and not callable(service):
            raise TypeError(f"Service '{name}' must be a class or callable")
        
        try:
            if factory:
                self._factories[name] = factory
            else:
                self._services[name] = service
            
            self._singleton_flags[name] = bool(singleton)
            
            # Store explicit dependencies or auto-detect from constructor
            if dependencies:
                if not isinstance(dependencies, list):
                    raise TypeError(f"Dependencies for service '{name}' must be a list")
                self._dependencies[name] = dependencies
            else:
                # Try to auto-detect dependencies from constructor
                if isinstance(service, type):
                    self._dependencies[name] = self._detect_dependencies(service)
                else:
                    self._dependencies[name] = []
            
            logger.debug(
                f"Registered service: {name} "
                f"(singleton={singleton}, deps={len(self._dependencies[name])})"
            )
        except Exception as e:
            logger.error(f"Error al registrar servicio '{name}': {e}", exc_info=True)
            raise
    
    def _detect_dependencies(self, service_class: Type) -> List[str]:
        """
        Auto-detect dependencies from constructor signature.
        
        Only includes required parameters (those without default values).
        Optional parameters are ignored as they have default values.
        
        Args:
            service_class: Class to inspect
        
        Returns:
            List of detected dependency names
        """
        try:
            sig = inspect.signature(service_class.__init__)
            deps = []
            for param_name, param in sig.parameters.items():
                if param_name not in ('self', 'args', 'kwargs'):
                    # Only include required parameters (no default value)
                    if param.default == inspect.Parameter.empty:
                        deps.append(param_name)
            return deps
        except Exception as e:
            logger.warning(f"Could not detect dependencies for {service_class}: {e}")
            return []
    
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a service instance.
        
        Args:
            name: Service name
            instance: Service instance
        """
        self._singletons[name] = instance
        self._singleton_flags[name] = True
        logger.debug(f"Registered instance: {name}")
    
    def get(self, name: str, scope: Optional[str] = None, **kwargs) -> Any:
        """
        Get a service by name with automatic dependency resolution and improved error handling.
        
        Args:
            name: Service name
            scope: Optional scope identifier for scoped instances
            **kwargs: Additional arguments for instantiation (overrides resolved deps)
        
        Returns:
            Service instance
        
        Raises:
            ValueError: If service not found or name is invalid
            RuntimeError: If circular dependency detected or error creating instance
        """
        # Validación de nombre
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Service name must be a non-empty string")
        
        name = name.strip()
        
        # Check for circular dependency
        if name in self._resolving:
            circular_path = " -> ".join(self._resolving) + f" -> {name}"
            logger.error(f"Circular dependency detected: {circular_path}")
            raise RuntimeError(
                f"Circular dependency detected: {name} is already being resolved. "
                f"Path: {circular_path}"
            )
        
        # Check scoped instances
        if scope:
            if not isinstance(scope, str) or not scope.strip():
                raise ValueError("Scope must be a non-empty string")
            scope = scope.strip()
            if scope in self._scoped and name in self._scoped[scope]:
                logger.debug(f"Retornando instancia scoped '{name}' del scope '{scope}'")
                return self._scoped[scope][name]
        
        # Check if singleton exists
        if name in self._singletons:
            logger.debug(f"Retornando instancia singleton '{name}'")
            return self._singletons[name]
        
        # Mark as resolving
        self._resolving.add(name)
        
        try:
            # Resolve dependencies
            resolved_deps = self._resolve_dependencies(name, scope)
            resolved_deps.update(kwargs)  # Override with explicit kwargs
            
            logger.debug(
                f"Creando instancia de '{name}' con {len(resolved_deps)} dependencias"
            )
            
            # Create instance
            try:
                if name in self._factories:
                    factory = self._factories[name]
                    # If factory is a callable that takes no args, call it directly
                    sig = inspect.signature(factory)
                    if not resolved_deps and not sig.parameters:
                        instance = factory()
                    else:
                        instance = factory(**resolved_deps)
                elif name in self._services:
                    service = self._services[name]
                    if isinstance(service, type):
                        instance = service(**resolved_deps)
                    else:
                        instance = service
                else:
                    available = list(self._services.keys()) + list(self._factories.keys()) + list(self._singletons.keys())
                    raise ValueError(
                        f"Service '{name}' not registered. "
                        f"Available services: {', '.join(available) if available else 'none'}"
                    )
            except Exception as e:
                logger.error(
                    f"Error al crear instancia de servicio '{name}': {e}",
                    exc_info=True
                )
                raise RuntimeError(f"Error al crear instancia de servicio '{name}': {e}") from e
            
            # Store instance
            if scope:
                if scope not in self._scoped:
                    self._scoped[scope] = {}
                self._scoped[scope][name] = instance
                logger.debug(f"Instancia scoped '{name}' almacenada en scope '{scope}'")
            elif self._singleton_flags.get(name, True):
                self._singletons[name] = instance
                logger.debug(f"Instancia singleton '{name}' almacenada")
            
            return instance
        finally:
            # Remove from resolving set
            self._resolving.discard(name)
    
    def _resolve_dependencies(self, service_name: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve dependencies for a service with improved error handling.
        
        Args:
            service_name: Name of the service
            scope: Optional scope identifier
        
        Returns:
            Dictionary of resolved dependencies
        """
        resolved = {}
        deps = self._dependencies.get(service_name, [])
        
        if not deps:
            logger.debug(f"Servicio '{service_name}' no tiene dependencias")
            return resolved
        
        logger.debug(f"Resolviendo {len(deps)} dependencias para '{service_name}': {deps}")
        
        missing_deps = []
        for dep_name in deps:
            try:
                resolved[dep_name] = self.get(dep_name, scope=scope)
                logger.debug(f"Dependencia '{dep_name}' resuelta para '{service_name}'")
            except ValueError as e:
                # Dependency not registered
                missing_deps.append(dep_name)
                logger.warning(
                    f"Dependencia '{dep_name}' no encontrada para servicio '{service_name}': {e}"
                )
                # Continue without this dependency - let the constructor handle it
            except RuntimeError as e:
                # Circular dependency or other runtime error
                logger.error(
                    f"Error al resolver dependencia '{dep_name}' para '{service_name}': {e}",
                    exc_info=True
                )
                # Re-raise runtime errors as they are critical
                raise
            except Exception as e:
                logger.error(
                    f"Error inesperado al resolver dependencia '{dep_name}' para '{service_name}': {e}",
                    exc_info=True
                )
                # Continue without this dependency - let the constructor handle it
        
        if missing_deps:
            logger.warning(
                f"Servicio '{service_name}' tiene {len(missing_deps)} dependencias faltantes: {missing_deps}"
            )
        
        logger.debug(
            f"Resueltas {len(resolved)}/{len(deps)} dependencias para '{service_name}'"
        )
        return resolved
    
    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services or name in self._singletons or name in self._factories
    
    def clear_scope(self, scope: str) -> None:
        """
        Clear all instances in a scope.
        
        Args:
            scope: Scope identifier
        """
        if scope in self._scoped:
            del self._scoped[scope]
            logger.debug(f"Cleared scope: {scope}")
    
    def clear(self) -> None:
        """Clear all registered services and instances."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_flags.clear()
        self._dependencies.clear()
        self._scoped.clear()
        self._resolving.clear()
        logger.debug("Cleared all services")


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container

