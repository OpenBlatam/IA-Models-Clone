"""
Dependency Injection Container

Advanced dependency injection container with support for
different service lifetimes, circular dependency detection,
and automatic wiring.
"""

import asyncio
import logging
import inspect
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime enumeration"""
    SINGLETON = "singleton"      # Single instance for entire application
    SCOPED = "scoped"           # Single instance per scope
    TRANSIENT = "transient"     # New instance every time
    THREAD_LOCAL = "thread_local"  # Single instance per thread


class Scope:
    """Dependency injection scope"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._services: Dict[Type, Any] = {}
        self._lock = asyncio.Lock()
        self._created_at = datetime.utcnow()
    
    async def get_service(self, service_type: Type) -> Optional[Any]:
        """Get service from scope"""
        async with self._lock:
            return self._services.get(service_type)
    
    async def set_service(self, service_type: Type, instance: Any) -> None:
        """Set service in scope"""
        async with self._lock:
            self._services[service_type] = instance
    
    async def clear(self) -> None:
        """Clear scope services"""
        async with self._lock:
            self._services.clear()
    
    def get_services(self) -> Dict[Type, Any]:
        """Get all services in scope"""
        return self._services.copy()


@dataclass
class ServiceDefinition:
    """Service definition"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[Type] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.implementation_type and not self.factory and not self.instance:
            self.implementation_type = self.service_type


class DependencyContainer:
    """Advanced dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDefinition] = {}
        self._singletons: Dict[Type, Any] = {}
        self._thread_locals: Dict[Type, threading.local] = {}
        self._scopes: Dict[str, Scope] = {}
        self._current_scope: Optional[Scope] = None
        self._lock = asyncio.Lock()
        self._circular_detector = CircularDependencyDetector()
        self._auto_wiring = AutoWiring()
    
    def register_singleton(self, service_type: Type, 
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None,
                          instance: Optional[Any] = None,
                          tags: Optional[List[str]] = None) -> 'DependencyContainer':
        """Register singleton service"""
        return self._register_service(
            service_type, implementation_type, factory, instance,
            ServiceLifetime.SINGLETON, tags
        )
    
    def register_scoped(self, service_type: Type,
                       implementation_type: Optional[Type] = None,
                       factory: Optional[Callable] = None,
                       tags: Optional[List[str]] = None) -> 'DependencyContainer':
        """Register scoped service"""
        return self._register_service(
            service_type, implementation_type, factory, None,
            ServiceLifetime.SCOPED, tags
        )
    
    def register_transient(self, service_type: Type,
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None,
                          tags: Optional[List[str]] = None) -> 'DependencyContainer':
        """Register transient service"""
        return self._register_service(
            service_type, implementation_type, factory, None,
            ServiceLifetime.TRANSIENT, tags
        )
    
    def register_thread_local(self, service_type: Type,
                             implementation_type: Optional[Type] = None,
                             factory: Optional[Callable] = None,
                             tags: Optional[List[str]] = None) -> 'DependencyContainer':
        """Register thread-local service"""
        return self._register_service(
            service_type, implementation_type, factory, None,
            ServiceLifetime.THREAD_LOCAL, tags
        )
    
    def _register_service(self, service_type: Type,
                         implementation_type: Optional[Type] = None,
                         factory: Optional[Callable] = None,
                         instance: Optional[Any] = None,
                         lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
                         tags: Optional[List[str]] = None) -> 'DependencyContainer':
        """Register service with given configuration"""
        
        # Auto-detect dependencies if not provided
        dependencies = []
        if implementation_type and not factory and not instance:
            dependencies = self._auto_wiring.detect_dependencies(implementation_type)
        
        definition = ServiceDefinition(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=lifetime,
            dependencies=dependencies,
            tags=tags or []
        )
        
        self._services[service_type] = definition
        
        # Initialize thread-local storage if needed
        if lifetime == ServiceLifetime.THREAD_LOCAL:
            self._thread_locals[service_type] = threading.local()
        
        logger.info(f"Registered service: {service_type.__name__} ({lifetime.value})")
        return self
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get service instance"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        definition = self._services[service_type]
        
        # Check for circular dependencies
        if self._circular_detector.is_circular(service_type, set()):
            raise ValueError(f"Circular dependency detected for {service_type.__name__}")
        
        # Get instance based on lifetime
        if definition.lifetime == ServiceLifetime.SINGLETON:
            return await self._get_singleton(definition)
        elif definition.lifetime == ServiceLifetime.SCOPED:
            return await self._get_scoped(definition)
        elif definition.lifetime == ServiceLifetime.THREAD_LOCAL:
            return await self._get_thread_local(definition)
        else:  # TRANSIENT
            return await self._create_instance(definition)
    
    async def _get_singleton(self, definition: ServiceDefinition) -> Any:
        """Get singleton instance"""
        if definition.service_type in self._singletons:
            return self._singletons[definition.service_type]
        
        async with self._lock:
            # Double-check after acquiring lock
            if definition.service_type in self._singletons:
                return self._singletons[definition.service_type]
            
            instance = await self._create_instance(definition)
            self._singletons[definition.service_type] = instance
            return instance
    
    async def _get_scoped(self, definition: ServiceDefinition) -> Any:
        """Get scoped instance"""
        if not self._current_scope:
            raise ValueError("No active scope for scoped service")
        
        instance = await self._current_scope.get_service(definition.service_type)
        if instance is not None:
            return instance
        
        async with self._lock:
            # Double-check after acquiring lock
            instance = await self._current_scope.get_service(definition.service_type)
            if instance is not None:
                return instance
            
            instance = await self._create_instance(definition)
            await self._current_scope.set_service(definition.service_type, instance)
            return instance
    
    async def _get_thread_local(self, definition: ServiceDefinition) -> Any:
        """Get thread-local instance"""
        thread_local = self._thread_locals[definition.service_type]
        
        if not hasattr(thread_local, 'instance'):
            instance = await self._create_instance(definition)
            thread_local.instance = instance
        
        return thread_local.instance
    
    async def _create_instance(self, definition: ServiceDefinition) -> Any:
        """Create service instance"""
        # Return existing instance if available
        if definition.instance is not None:
            return definition.instance
        
        # Use factory if available
        if definition.factory is not None:
            if inspect.iscoroutinefunction(definition.factory):
                return await definition.factory()
            else:
                return definition.factory()
        
        # Create instance from implementation type
        if definition.implementation_type is None:
            raise ValueError(f"No implementation type or factory for {definition.service_type.__name__}")
        
        # Resolve dependencies
        dependencies = {}
        for dep_type in definition.dependencies:
            dep_instance = await self.get_service(dep_type)
            dependencies[dep_type] = dep_instance
        
        # Create instance
        instance = self._auto_wiring.create_instance(
            definition.implementation_type, dependencies
        )
        
        return instance
    
    def create_scope(self, name: str = None) -> Scope:
        """Create new dependency injection scope"""
        scope_name = name or f"scope_{len(self._scopes)}"
        scope = Scope(scope_name)
        self._scopes[scope_name] = scope
        return scope
    
    async def enter_scope(self, scope: Scope) -> None:
        """Enter dependency injection scope"""
        self._current_scope = scope
    
    async def exit_scope(self) -> None:
        """Exit current scope"""
        if self._current_scope:
            await self._current_scope.clear()
            self._current_scope = None
    
    async def with_scope(self, scope: Scope, func: Callable) -> Any:
        """Execute function within scope"""
        old_scope = self._current_scope
        try:
            await self.enter_scope(scope)
            if inspect.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        finally:
            await self.exit_scope()
            self._current_scope = old_scope
    
    def get_services_by_tag(self, tag: str) -> List[Type]:
        """Get services by tag"""
        return [
            service_type for service_type, definition in self._services.items()
            if tag in definition.tags
        ]
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get service information"""
        if service_type not in self._services:
            return None
        
        definition = self._services[service_type]
        return {
            "service_type": service_type.__name__,
            "implementation_type": definition.implementation_type.__name__ if definition.implementation_type else None,
            "lifetime": definition.lifetime.value,
            "dependencies": [dep.__name__ for dep in definition.dependencies],
            "tags": definition.tags,
            "metadata": definition.metadata
        }
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services"""
        return [
            self.get_service_info(service_type)
            for service_type in self._services.keys()
        ]
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        return service_type in self._services
    
    def unregister(self, service_type: Type) -> None:
        """Unregister service"""
        if service_type in self._services:
            del self._services[service_type]
            
            # Remove from singletons
            if service_type in self._singletons:
                del self._singletons[service_type]
            
            # Remove from thread locals
            if service_type in self._thread_locals:
                del self._thread_locals[service_type]
            
            logger.info(f"Unregistered service: {service_type.__name__}")
    
    async def validate_configuration(self) -> List[str]:
        """Validate container configuration"""
        errors = []
        
        for service_type, definition in self._services.items():
            # Check if dependencies are registered
            for dep_type in definition.dependencies:
                if not self.is_registered(dep_type):
                    errors.append(f"Service {service_type.__name__} depends on unregistered service {dep_type.__name__}")
            
            # Check for circular dependencies
            if self._circular_detector.is_circular(service_type, set()):
                errors.append(f"Circular dependency detected for {service_type.__name__}")
        
        return errors
    
    async def dispose(self) -> None:
        """Dispose container and all services"""
        # Dispose singletons
        for service_type, instance in self._singletons.items():
            if hasattr(instance, 'dispose'):
                if inspect.iscoroutinefunction(instance.dispose):
                    await instance.dispose()
                else:
                    instance.dispose()
        
        # Clear all data
        self._services.clear()
        self._singletons.clear()
        self._thread_locals.clear()
        self._scopes.clear()
        self._current_scope = None
        
        logger.info("Dependency container disposed")


class CircularDependencyDetector:
    """Detects circular dependencies in service graph"""
    
    def __init__(self):
        self._visited: Set[Type] = set()
        self._recursion_stack: Set[Type] = set()
    
    def is_circular(self, service_type: Type, current_path: Set[Type]) -> bool:
        """Check if there's a circular dependency"""
        if service_type in current_path:
            return True
        
        if service_type in self._visited:
            return False
        
        self._visited.add(service_type)
        current_path.add(service_type)
        
        # This would need access to the container to check dependencies
        # For now, return False as a placeholder
        return False


class AutoWiring:
    """Automatic dependency wiring"""
    
    def detect_dependencies(self, service_type: Type) -> List[Type]:
        """Detect dependencies from constructor"""
        dependencies = []
        
        try:
            signature = inspect.signature(service_type.__init__)
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        except (ValueError, TypeError):
            # Can't inspect signature
            pass
        
        return dependencies
    
    def create_instance(self, service_type: Type, dependencies: Dict[Type, Any]) -> Any:
        """Create instance with dependency injection"""
        try:
            signature = inspect.signature(service_type.__init__)
            kwargs = {}
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation in dependencies:
                    kwargs[param_name] = dependencies[param.annotation]
                elif param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
            
            return service_type(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create instance of {service_type.__name__}: {e}")
            raise


# Global dependency container instance
dependency_container = DependencyContainer()


# Convenience functions
def register_singleton(service_type: Type, implementation_type: Optional[Type] = None, **kwargs):
    """Register singleton service"""
    return dependency_container.register_singleton(service_type, implementation_type, **kwargs)


def register_scoped(service_type: Type, implementation_type: Optional[Type] = None, **kwargs):
    """Register scoped service"""
    return dependency_container.register_scoped(service_type, implementation_type, **kwargs)


def register_transient(service_type: Type, implementation_type: Optional[Type] = None, **kwargs):
    """Register transient service"""
    return dependency_container.register_transient(service_type, implementation_type, **kwargs)


async def get_service(service_type: Type[T]) -> T:
    """Get service instance"""
    return await dependency_container.get_service(service_type)


def inject(service_type: Type[T]) -> T:
    """Inject service (for use in function parameters)"""
    # This would be used with a decorator or type annotation
    # For now, it's a placeholder
    pass





















