"""
Dependency Injection Container
=============================

Advanced dependency injection container with support for:
- Singleton and transient lifetimes
- Interface-based registration
- Async factory functions
- Circular dependency detection
- Configuration-based setup
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Type, TypeVar, Callable, Optional, Union, List
import asyncio
import inspect
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifetime(Enum):
    """Service lifetime options"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceRegistration:
    """Service registration information"""
    service_type: Type[Any]
    implementation_type: Optional[Type[Any]] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: Lifetime = Lifetime.TRANSIENT
    dependencies: List[Type[Any]] = field(default_factory=list)
    is_async: bool = False


class DependencyInjectionContainer:
    """
    Advanced dependency injection container
    
    Supports singleton, transient, and scoped lifetimes with
    automatic dependency resolution and circular dependency detection.
    """
    
    def __init__(self):
        self._registrations: Dict[Type[Any], ServiceRegistration] = {}
        self._singletons: Dict[Type[Any], Any] = {}
        self._scoped_instances: Dict[Type[Any], Any] = {}
        self._resolution_stack: List[Type[Any]] = []
        self._lock = asyncio.Lock()
    
    def register_singleton(
        self, 
        service_type: Type[T], 
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None
    ) -> None:
        """Register a singleton service"""
        self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=Lifetime.SINGLETON
        )
    
    def register_transient(
        self, 
        service_type: Type[T], 
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None
    ) -> None:
        """Register a transient service"""
        self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=Lifetime.TRANSIENT
        )
    
    def register_scoped(
        self, 
        service_type: Type[T], 
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None
    ) -> None:
        """Register a scoped service"""
        self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=Lifetime.SCOPED
        )
    
    def _register_service(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable] = None,
        instance: Optional[T] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> None:
        """Register a service with the container"""
        if service_type in self._registrations:
            logger.warning(f"Service {service_type.__name__} is already registered. Overwriting.")
        
        # Determine dependencies
        dependencies = []
        if implementation_type:
            dependencies = self._get_constructor_dependencies(implementation_type)
        elif factory:
            dependencies = self._get_factory_dependencies(factory)
        
        # Check if factory is async
        is_async = False
        if factory and inspect.iscoroutinefunction(factory):
            is_async = True
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=lifetime,
            dependencies=dependencies,
            is_async=is_async
        )
        
        self._registrations[service_type] = registration
        logger.debug(f"Registered {lifetime.value} service: {service_type.__name__}")
    
    def _get_constructor_dependencies(self, implementation_type: Type[Any]) -> List[Type[Any]]:
        """Get constructor dependencies for a type"""
        try:
            signature = inspect.signature(implementation_type.__init__)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
        except Exception as e:
            logger.warning(f"Could not get dependencies for {implementation_type.__name__}: {e}")
            return []
    
    def _get_factory_dependencies(self, factory: Callable) -> List[Type[Any]]:
        """Get factory function dependencies"""
        try:
            signature = inspect.signature(factory)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
        except Exception as e:
            logger.warning(f"Could not get dependencies for factory {factory.__name__}: {e}")
            return []
    
    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance"""
        async with self._lock:
            return await self._resolve_service(service_type)
    
    async def _resolve_service(self, service_type: Type[T]) -> T:
        """Resolve a service instance (internal method)"""
        # Check for circular dependencies
        if service_type in self._resolution_stack:
            stack_str = " -> ".join([t.__name__ for t in self._resolution_stack + [service_type]])
            raise CircularDependencyError(f"Circular dependency detected: {stack_str}")
        
        # Check if service is registered
        if service_type not in self._registrations:
            raise ServiceNotRegisteredError(f"Service {service_type.__name__} is not registered")
        
        registration = self._registrations[service_type]
        
        # Handle different lifetimes
        if registration.lifetime == Lifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = await self._create_instance(registration)
            self._singletons[service_type] = instance
            return instance
        
        elif registration.lifetime == Lifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = await self._create_instance(registration)
            self._scoped_instances[service_type] = instance
            return instance
        
        else:  # TRANSIENT
            return await self._create_instance(registration)
    
    async def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create an instance of a service"""
        # Add to resolution stack
        self._resolution_stack.append(registration.service_type)
        
        try:
            # If instance is provided, return it
            if registration.instance is not None:
                return registration.instance
            
            # Resolve dependencies
            dependencies = []
            for dep_type in registration.dependencies:
                dep_instance = await self._resolve_service(dep_type)
                dependencies.append(dep_instance)
            
            # Create instance
            if registration.factory:
                if registration.is_async:
                    instance = await registration.factory(*dependencies)
                else:
                    instance = registration.factory(*dependencies)
            elif registration.implementation_type:
                instance = registration.implementation_type(*dependencies)
            else:
                raise ServiceRegistrationError(
                    f"Service {registration.service_type.__name__} has no implementation"
                )
            
            return instance
            
        finally:
            # Remove from resolution stack
            self._resolution_stack.pop()
    
    def resolve_sync(self, service_type: Type[T]) -> T:
        """Resolve a service instance synchronously"""
        return asyncio.run(self.resolve(service_type))
    
    def is_registered(self, service_type: Type[Any]) -> bool:
        """Check if a service is registered"""
        return service_type in self._registrations
    
    def get_registration(self, service_type: Type[Any]) -> Optional[ServiceRegistration]:
        """Get service registration information"""
        return self._registrations.get(service_type)
    
    def clear_scoped_instances(self) -> None:
        """Clear all scoped instances"""
        self._scoped_instances.clear()
        logger.debug("Cleared scoped instances")
    
    def clear_singletons(self) -> None:
        """Clear all singleton instances"""
        self._singletons.clear()
        logger.debug("Cleared singleton instances")
    
    def clear_all(self) -> None:
        """Clear all registrations and instances"""
        self._registrations.clear()
        self._singletons.clear()
        self._scoped_instances.clear()
        self._resolution_stack.clear()
        logger.debug("Cleared all container data")
    
    @asynccontextmanager
    async def scope(self):
        """Create a new scope for scoped services"""
        old_scoped = self._scoped_instances.copy()
        self._scoped_instances.clear()
        
        try:
            yield self
        finally:
            self._scoped_instances = old_scoped
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get container statistics"""
        return {
            "total_registrations": len(self._registrations),
            "singleton_instances": len(self._singletons),
            "scoped_instances": len(self._scoped_instances),
            "registrations_by_lifetime": {
                lifetime.value: sum(
                    1 for reg in self._registrations.values() 
                    if reg.lifetime == lifetime
                )
                for lifetime in Lifetime
            }
        }


class CircularDependencyError(Exception):
    """Exception raised when circular dependency is detected"""
    pass


class ServiceNotRegisteredError(Exception):
    """Exception raised when trying to resolve unregistered service"""
    pass


class ServiceRegistrationError(Exception):
    """Exception raised when service registration is invalid"""
    pass


# Global container instance
_container: Optional[DependencyInjectionContainer] = None


def get_container() -> DependencyInjectionContainer:
    """Get the global container instance"""
    global _container
    if _container is None:
        _container = DependencyInjectionContainer()
    return _container


def configure_container() -> DependencyInjectionContainer:
    """Configure the global container with default services"""
    container = get_container()
    
    # Register container itself
    container.register_singleton(DependencyInjectionContainer, instance=container)
    
    return container


# Decorator for automatic dependency injection
def inject(service_type: Type[T]) -> T:
    """Decorator for automatic dependency injection"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            container = get_container()
            service = await container.resolve(service_type)
            return await func(service, *args, **kwargs)
        return wrapper
    return decorator




