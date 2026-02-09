#!/usr/bin/env python3
"""
Dependency Container for Video-OpusClip
Dependency injection container with service lifetime management
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import weakref
import inspect
from functools import wraps

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime enumeration"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Service descriptor for dependency injection"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    instance: Optional[Any] = None
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceProvider:
    """Service provider for dependency resolution"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._scope_stack: List[Dict[Type, Any]] = []
    
    def register_service(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        **metadata
    ) -> 'ServiceProvider':
        """Register a service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=lifetime,
            metadata=metadata
        )
        
        # Analyze dependencies
        if implementation_type:
            descriptor.dependencies = self._analyze_dependencies(implementation_type)
        elif factory:
            descriptor.dependencies = self._analyze_factory_dependencies(factory)
        
        self._services[service_type] = descriptor
        return self
    
    def _analyze_dependencies(self, service_type: Type) -> List[Type]:
        """Analyze constructor dependencies"""
        dependencies = []
        
        if hasattr(service_type, '__init__'):
            sig = inspect.signature(service_type.__init__)
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        
        return dependencies
    
    def _analyze_factory_dependencies(self, factory: Callable) -> List[Type]:
        """Analyze factory function dependencies"""
        dependencies = []
        sig = inspect.signature(factory)
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                dependencies.append(param.annotation)
        
        return dependencies
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")
        
        descriptor = self._services[service_type]
        
        # Check if instance already exists for singleton
        if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance is not None:
            return descriptor.instance
        
        # Check if instance exists in current scope
        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if self._scope_stack and service_type in self._scope_stack[-1]:
                return self._scope_stack[-1][service_type]
        
        # Create new instance
        instance = self._create_instance(descriptor)
        
        # Store instance based on lifetime
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            descriptor.instance = instance
        elif descriptor.lifetime == ServiceLifetime.SCOPED and self._scope_stack:
            self._scope_stack[-1][service_type] = instance
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a service instance"""
        if descriptor.factory:
            # Use factory function
            dependencies = [self.resolve(dep) for dep in descriptor.dependencies]
            return descriptor.factory(*dependencies)
        elif descriptor.implementation_type:
            # Use implementation type
            dependencies = [self.resolve(dep) for dep in descriptor.dependencies]
            return descriptor.implementation_type(*dependencies)
        else:
            # Use service type directly
            dependencies = [self.resolve(dep) for dep in descriptor.dependencies]
            return descriptor.service_type(*dependencies)
    
    def resolve_all(self, service_type: Type[T]) -> List[T]:
        """Resolve all services of a given type"""
        instances = []
        for registered_type, descriptor in self._services.items():
            if issubclass(registered_type, service_type):
                instances.append(self.resolve(registered_type))
        return instances
    
    @asynccontextmanager
    async def create_scope(self):
        """Create a new service scope"""
        scope = {}
        self._scope_stack.append(scope)
        try:
            yield scope
        finally:
            self._scope_stack.pop()
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services"""
        return self._services.copy()
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered"""
        return service_type in self._services


class DependencyContainer:
    """Main dependency injection container"""
    
    def __init__(self):
        self._service_provider = ServiceProvider()
        self._configuration: Dict[str, Any] = {}
    
    def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        **metadata
    ) -> 'DependencyContainer':
        """Register a singleton service"""
        self._service_provider.register_service(
            service_type, implementation_type, factory, ServiceLifetime.SINGLETON, **metadata
        )
        return self
    
    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        **metadata
    ) -> 'DependencyContainer':
        """Register a transient service"""
        self._service_provider.register_service(
            service_type, implementation_type, factory, ServiceLifetime.TRANSIENT, **metadata
        )
        return self
    
    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        **metadata
    ) -> 'DependencyContainer':
        """Register a scoped service"""
        self._service_provider.register_service(
            service_type, implementation_type, factory, ServiceLifetime.SCOPED, **metadata
        )
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service"""
        return self._service_provider.resolve(service_type)
    
    def resolve_all(self, service_type: Type[T]) -> List[T]:
        """Resolve all services of a given type"""
        return self._service_provider.resolve_all(service_type)
    
    def create_scope(self):
        """Create a new service scope"""
        return self._service_provider.create_scope()
    
    def configure(self, key: str, value: Any) -> 'DependencyContainer':
        """Configure the container"""
        self._configuration[key] = value
        return self
    
    def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._configuration.get(key, default)
    
    def get_service_provider(self) -> ServiceProvider:
        """Get the underlying service provider"""
        return self._service_provider
    
    def validate(self) -> List[str]:
        """Validate the container configuration"""
        errors = []
        
        for service_type, descriptor in self._service_provider.get_registered_services().items():
            # Check for circular dependencies
            if self._has_circular_dependency(service_type, set()):
                errors.append(f"Circular dependency detected for {service_type.__name__}")
            
            # Check if dependencies are registered
            for dep in descriptor.dependencies:
                if not self._service_provider.is_registered(dep):
                    errors.append(f"Dependency {dep.__name__} for {service_type.__name__} is not registered")
        
        return errors
    
    def _has_circular_dependency(self, service_type: Type, visited: set) -> bool:
        """Check for circular dependencies"""
        if service_type in visited:
            return True
        
        if service_type not in self._service_provider.get_registered_services():
            return False
        
        visited.add(service_type)
        descriptor = self._service_provider.get_registered_services()[service_type]
        
        for dep in descriptor.dependencies:
            if self._has_circular_dependency(dep, visited.copy()):
                return True
        
        return False


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get the global dependency container"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def register_singleton(
    service_type: Type[T],
    implementation_type: Optional[Type[T]] = None,
    factory: Optional[Callable[..., T]] = None,
    **metadata
) -> DependencyContainer:
    """Register a singleton service globally"""
    return get_container().register_singleton(service_type, implementation_type, factory, **metadata)


def register_transient(
    service_type: Type[T],
    implementation_type: Optional[Type[T]] = None,
    factory: Optional[Callable[..., T]] = None,
    **metadata
) -> DependencyContainer:
    """Register a transient service globally"""
    return get_container().register_transient(service_type, implementation_type, factory, **metadata)


def register_scoped(
    service_type: Type[T],
    implementation_type: Optional[Type[T]] = None,
    factory: Optional[Callable[..., T]] = None,
    **metadata
) -> DependencyContainer:
    """Register a scoped service globally"""
    return get_container().register_scoped(service_type, implementation_type, factory, **metadata)


def resolve(service_type: Type[T]) -> T:
    """Resolve a service globally"""
    return get_container().resolve(service_type)


def resolve_all(service_type: Type[T]) -> List[T]:
    """Resolve all services of a given type globally"""
    return get_container().resolve_all(service_type)


def get_service_provider() -> ServiceProvider:
    """Get the global service provider"""
    return get_container().get_service_provider()


# Dependency injection decorator
def inject(*dependencies: Type):
    """Decorator for dependency injection"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            resolved_deps = [container.resolve(dep) for dep in dependencies]
            return func(*args, *resolved_deps, **kwargs)
        return wrapper
    return decorator


# Async dependency injection decorator
def inject_async(*dependencies: Type):
    """Decorator for async dependency injection"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            container = get_container()
            resolved_deps = [container.resolve(dep) for dep in dependencies]
            return await func(*args, *resolved_deps, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Example service classes
    class ILogger:
        def log(self, message: str):
            pass
    
    class ConsoleLogger(ILogger):
        def log(self, message: str):
            print(f"[CONSOLE] {message}")
    
    class FileLogger(ILogger):
        def __init__(self, filename: str):
            self.filename = filename
        
        def log(self, message: str):
            print(f"[FILE:{self.filename}] {message}")
    
    class EmailService:
        def __init__(self, logger: ILogger):
            self.logger = logger
        
        def send_email(self, to: str, subject: str):
            self.logger.log(f"Sending email to {to}: {subject}")
    
    # Register services
    container = get_container()
    container.register_singleton(ILogger, ConsoleLogger)
    container.register_transient(EmailService)
    
    # Resolve and use services
    email_service = container.resolve(EmailService)
    email_service.send_email("user@example.com", "Hello!")
    
    # Use decorators
    @inject(ILogger)
    def send_notification(logger: ILogger, message: str):
        logger.log(f"Notification: {message}")
    
    send_notification("Test message")
    
    print("✅ Dependency container example completed!") 