"""
Dependency Container (Mejorado)

Centralized dependency injection container following Inversion of Control principles.
Supports multiple scopes (singleton, scoped, transient) with automatic dependency resolution.

Features:
- Singleton: One instance for the entire application lifetime
- Scoped: One instance per request/scope
- Transient: New instance every time
- Automatic dependency resolution based on type hints
- Thread-safe operations
- Async support
"""

from typing import Optional, Dict, Any, Callable, TypeVar, Type, List, Union
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.orm import Session
import asyncio
import inspect
from functools import wraps

T = TypeVar('T')


class ServiceScope(Enum):
    """Service lifecycle scopes"""
    SINGLETON = "singleton"  # One instance for application lifetime
    SCOPED = "scoped"        # One instance per scope (e.g., request)
    TRANSIENT = "transient"   # New instance every time


@dataclass
class ServiceRegistration:
    """Registration information for a service"""
    factory: Callable[..., Any]
    scope: ServiceScope = ServiceScope.SINGLETON
    instance: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    service_type: Optional[Type] = None
    is_async: bool = False


class DependencyContainer:
    """
    Advanced dependency injection container.
    
    Manages lifecycle and dependencies of application components with support for:
    - Multiple scopes (singleton, scoped, transient)
    - Automatic dependency resolution
    - Thread-safe operations
    - Async support
    
    Example:
        container = DependencyContainer()
        
        # Register singleton
        container.register_singleton('ranking_service', RankingService)
        
        # Register scoped (per request)
        container.register_scoped('chat_service', ChatService, dependencies=['chat_repository'])
        
        # Get service
        service = await container.get('chat_service')
    """
    
    _instance: Optional['DependencyContainer'] = None
    
    def __new__(cls):
        """Singleton pattern implementation for container instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize container state"""
        self._services: Dict[str, ServiceRegistration] = {}
        self._type_registry: Dict[Type, str] = {}  # Map type to service name
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}  # scope_id -> {service_name: instance}
        self._lock = asyncio.Lock()
        self._db_session: Optional[Session] = None
    
    def register_singleton(
        self,
        service_name: str,
        factory: Union[Callable[..., T], Type[T]],
        dependencies: Optional[List[str]] = None,
        service_type: Optional[Type] = None
    ) -> None:
        """
        Register a service as singleton.
        
        Args:
            service_name: Unique name for the service
            factory: Factory function or class to create instances
            dependencies: List of dependency service names (optional, auto-detected if None)
            service_type: Type of the service (for type-based resolution)
        """
        self._register_service(
            service_name=service_name,
            factory=factory,
            scope=ServiceScope.SINGLETON,
            dependencies=dependencies,
            service_type=service_type
        )
    
    def register_scoped(
        self,
        service_name: str,
        factory: Union[Callable[..., T], Type[T]],
        dependencies: Optional[List[str]] = None,
        service_type: Optional[Type] = None
    ) -> None:
        """
        Register a service with scoped lifetime (one per scope/request).
        
        Args:
            service_name: Unique name for the service
            factory: Factory function or class to create instances
            dependencies: List of dependency service names (optional, auto-detected if None)
            service_type: Type of the service (for type-based resolution)
        """
        self._register_service(
            service_name=service_name,
            factory=factory,
            scope=ServiceScope.SCOPED,
            dependencies=dependencies,
            service_type=service_type
        )
    
    def register_transient(
        self,
        service_name: str,
        factory: Union[Callable[..., T], Type[T]],
        dependencies: Optional[List[str]] = None,
        service_type: Optional[Type] = None
    ) -> None:
        """
        Register a service as transient (new instance every time).
        
        Args:
            service_name: Unique name for the service
            factory: Factory function or class to create instances
            dependencies: List of dependency service names (optional, auto-detected if None)
            service_type: Type of the service (for type-based resolution)
        """
        self._register_service(
            service_name=service_name,
            factory=factory,
            scope=ServiceScope.TRANSIENT,
            dependencies=dependencies,
            service_type=service_type
        )
    
    def _register_service(
        self,
        service_name: str,
        factory: Union[Callable[..., T], Type[T]],
        scope: ServiceScope,
        dependencies: Optional[List[str]] = None,
        service_type: Optional[Type] = None
    ) -> None:
        """Internal method to register a service"""
        # Auto-detect dependencies from type hints if not provided
        if dependencies is None:
            dependencies = self._extract_dependencies(factory)
        
        # Determine service type
        if service_type is None:
            if inspect.isclass(factory):
                service_type = factory
            else:
                # Try to infer from return type annotation
                sig = inspect.signature(factory)
                return_annotation = sig.return_annotation
                if return_annotation != inspect.Signature.empty:
                    service_type = return_annotation
        
        # Check if async
        is_async = inspect.iscoroutinefunction(factory) or (
            inspect.isclass(factory) and hasattr(factory, '__init__') and
            inspect.iscoroutinefunction(factory.__init__)
        )
        
        registration = ServiceRegistration(
            factory=factory,
            scope=scope,
            dependencies=dependencies or [],
            service_type=service_type,
            is_async=is_async
        )
        
        self._services[service_name] = registration
        
        # Register type mapping for type-based resolution
        if service_type:
            self._type_registry[service_type] = service_name
    
    def _extract_dependencies(self, factory: Callable) -> List[str]:
        """Extract dependencies from factory function signature"""
        dependencies = []
        try:
            sig = inspect.signature(factory)
            for param_name, param in sig.parameters.items():
                # Skip self, cls, and parameters without type hints
                if param_name in ('self', 'cls'):
                    continue
                
                param_type = param.annotation
                if param_type != inspect.Parameter.empty:
                    # Try to find registered service by type
                    service_name = self._type_registry.get(param_type)
                    if service_name:
                        dependencies.append(service_name)
        except (ValueError, TypeError):
            # If signature inspection fails, return empty list
            pass
        return dependencies
    
    async def get(self, service_name: str, scope_id: Optional[str] = None) -> Any:
        """
        Get service instance by name.
        
        Args:
            service_name: Name of the service
            scope_id: Optional scope ID for scoped services (defaults to 'default')
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
            RuntimeError: If service creation fails
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not registered")
        
        registration = self._services[service_name]
        scope_id = scope_id or 'default'
        
        async with self._lock:
            # Handle different scopes
            if registration.scope == ServiceScope.SINGLETON:
                if registration.instance is None:
                    registration.instance = await self._create_instance(
                        registration, scope_id
                    )
                return registration.instance
            
            elif registration.scope == ServiceScope.SCOPED:
                if scope_id not in self._scoped_instances:
                    self._scoped_instances[scope_id] = {}
                
                if service_name not in self._scoped_instances[scope_id]:
                    self._scoped_instances[scope_id][service_name] = await self._create_instance(
                        registration, scope_id
                    )
                
                return self._scoped_instances[scope_id][service_name]
            
            else:  # TRANSIENT
                return await self._create_instance(registration, scope_id)
    
    def get_sync(self, service_name: str, scope_id: Optional[str] = None) -> Any:
        """
        Synchronous version of get().
        
        Note: Use only when you're sure you're in an async context or event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self.get(service_name, scope_id),
                    loop
                )
                return future.result()
            else:
                return loop.run_until_complete(self.get(service_name, scope_id))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.get(service_name, scope_id))
    
    async def get_by_type(self, service_type: Type[T], scope_id: Optional[str] = None) -> T:
        """
        Get service instance by type.
        
        Args:
            service_type: Type of the service
            scope_id: Optional scope ID for scoped services
            
        Returns:
            Service instance of the specified type
            
        Raises:
            KeyError: If service type not registered
        """
        service_name = self._type_registry.get(service_type)
        if not service_name:
            raise KeyError(f"Service type '{service_type.__name__}' not registered")
        return await self.get(service_name, scope_id)
    
    async def _create_instance(self, registration: ServiceRegistration, scope_id: str) -> Any:
        """Create a new service instance with dependency injection"""
        # Resolve dependencies
        deps = {}
        for dep_name in registration.dependencies:
            deps[dep_name] = await self.get(dep_name, scope_id)
        
        # Get factory
        factory = registration.factory
        
        # Create instance
        try:
            if registration.is_async:
                if inspect.isclass(factory):
                    # Async class constructor
                    instance = await factory(**deps)
                else:
                    # Async factory function
                    instance = await factory(**deps)
            else:
                if inspect.isclass(factory):
                    # Regular class constructor
                    instance = factory(**deps)
                else:
                    # Regular factory function
                    instance = factory(**deps)
            
            return instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to create instance of service '{factory.__name__}': {e}"
            ) from e
    
    def clear_scope(self, scope_id: str) -> None:
        """
        Clear all scoped instances for a given scope.
        
        Useful for cleanup after request completion.
        
        Args:
            scope_id: Scope ID to clear
        """
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
    
    def set_db_session(self, session: Session) -> None:
        """Set database session (for backward compatibility)"""
        self._db_session = session
    
    def get_db_session(self) -> Optional[Session]:
        """Get database session (for backward compatibility)"""
        return self._db_session
    
    # Backward compatibility methods
    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """Register factory (backward compatibility - registers as singleton)"""
        self.register_singleton(name, factory)
    
    def register_singleton_instance(self, name: str, instance: Any) -> None:
        """Register singleton instance (backward compatibility)"""
        if name not in self._services:
            # Create a registration for the instance
            registration = ServiceRegistration(
                factory=lambda: instance,
                scope=ServiceScope.SINGLETON,
                instance=instance
            )
            self._services[name] = registration
        else:
            self._services[name].instance = instance
    
    def clear(self) -> None:
        """Clear all registered services and scoped instances"""
        self._services.clear()
        self._type_registry.clear()
        self._scoped_instances.clear()
        self._db_session = None


# Global container instance
container = DependencyContainer()


# Decorators for automatic registration
def singleton(service_name: Optional[str] = None, service_type: Optional[Type] = None):
    """
    Decorator to automatically register a class as singleton.
    
    Example:
        @singleton('chat_service')
        class ChatService:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        name = service_name or cls.__name__.lower()
        container.register_singleton(name, cls, service_type=service_type or cls)
        return cls
    return decorator


def scoped(service_name: Optional[str] = None, service_type: Optional[Type] = None):
    """
    Decorator to automatically register a class as scoped.
    
    Example:
        @scoped('chat_service')
        class ChatService:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        name = service_name or cls.__name__.lower()
        container.register_scoped(name, cls, service_type=service_type or cls)
        return cls
    return decorator


def transient(service_name: Optional[str] = None, service_type: Optional[Type] = None):
    """
    Decorator to automatically register a class as transient.
    
    Example:
        @transient('chat_service')
        class ChatService:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        name = service_name or cls.__name__.lower()
        container.register_transient(name, cls, service_type=service_type or cls)
        return cls
    return decorator






