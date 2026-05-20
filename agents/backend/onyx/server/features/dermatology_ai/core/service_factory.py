"""
Service Factory for Dependency Injection
Creates and manages service instances with proper lifecycle
"""

from typing import Dict, Type, Any, Optional, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ServiceScope(str, Enum):
    """Service lifecycle scopes"""
    SINGLETON = "singleton"  # One instance for entire application
    REQUEST = "request"      # One instance per request
    TRANSIENT = "transient"  # New instance every time


@dataclass
class ServiceDescriptor:
    """Service registration descriptor"""
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    dependencies: List[str] = None
    config: Optional[Dict[str, Any]] = None


class ServiceFactory:
    """
    Service factory for dependency injection.
    Manages service creation and lifecycle.
    """
    
    def __init__(self):
        self.registrations: Dict[str, ServiceDescriptor] = {}
        self.singletons: Dict[str, Any] = {}
        self.request_instances: Dict[str, Any] = {}
    
    def register(
        self,
        service_name: str,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        scope: ServiceScope = ServiceScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Register a service
        
        Args:
            service_name: Unique service name
            service_type: Service interface/type
            implementation: Concrete implementation (if different from type)
            factory: Factory function to create instance
            scope: Service lifecycle scope
            dependencies: List of dependency service names
            config: Service configuration
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            factory=factory,
            scope=scope,
            dependencies=dependencies or [],
            config=config
        )
        
        self.registrations[service_name] = descriptor
        logger.debug(f"Registered service: {service_name} ({scope.value})")
    
    def register_singleton(
        self,
        service_name: str,
        service_type: Type,
        instance: Any
    ):
        """Register a singleton instance"""
        self.singletons[service_name] = instance
        self.register(
            service_name,
            service_type,
            scope=ServiceScope.SINGLETON
        )
        logger.debug(f"Registered singleton instance: {service_name}")
    
    async def create(self, service_name: str, **kwargs) -> Any:
        """
        Create or get service instance
        
        Args:
            service_name: Service name
            **kwargs: Additional arguments for factory
            
        Returns:
            Service instance
        """
        if service_name not in self.registrations:
            raise ValueError(f"Service {service_name} not registered")
        
        descriptor = self.registrations[service_name]
        
        # Check scope
        if descriptor.scope == ServiceScope.SINGLETON:
            if service_name in self.singletons:
                return self.singletons[service_name]
            
            instance = await self._create_instance(descriptor, **kwargs)
            self.singletons[service_name] = instance
            return instance
        
        elif descriptor.scope == ServiceScope.REQUEST:
            if service_name in self.request_instances:
                return self.request_instances[service_name]
            
            instance = await self._create_instance(descriptor, **kwargs)
            self.request_instances[service_name] = instance
            return instance
        
        else:  # TRANSIENT
            return await self._create_instance(descriptor, **kwargs)
    
    async def _create_instance(
        self,
        descriptor: ServiceDescriptor,
        **kwargs
    ) -> Any:
        """Create service instance"""
        # Resolve dependencies
        dependencies = {}
        for dep_name in descriptor.dependencies:
            dependencies[dep_name] = await self.create(dep_name)
        
        # Merge with kwargs
        all_kwargs = {**descriptor.config or {}, **dependencies, **kwargs}
        
        # Create instance
        if descriptor.factory:
            if callable(descriptor.factory):
                # Check if factory is async
                import inspect
                if inspect.iscoroutinefunction(descriptor.factory):
                    instance = await descriptor.factory(**all_kwargs)
                else:
                    instance = descriptor.factory(**all_kwargs)
            else:
                instance = descriptor.factory
        else:
            # Direct instantiation
            instance = descriptor.implementation(**all_kwargs)
            
            # Initialize if async
            if hasattr(instance, "initialize") and callable(instance.initialize):
                import inspect
                if inspect.iscoroutinefunction(instance.initialize):
                    await instance.initialize(descriptor.config)
                else:
                    instance.initialize(descriptor.config)
        
        return instance
    
    def clear_request_scope(self):
        """Clear request-scoped instances (call at end of request)"""
        # Cleanup request-scoped services
        for service_name, instance in self.request_instances.items():
            if hasattr(instance, "cleanup") and callable(instance.cleanup):
                import inspect
                if inspect.iscoroutinefunction(instance.cleanup):
                    import asyncio
                    asyncio.create_task(instance.cleanup())
                else:
                    instance.cleanup()
        
        self.request_instances.clear()
    
    def get_registered_services(self) -> List[str]:
        """Get list of registered service names"""
        return list(self.registrations.keys())
    
    def is_registered(self, service_name: str) -> bool:
        """Check if service is registered"""
        return service_name in self.registrations


# Global service factory
_service_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get or create global service factory"""
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
    return _service_factory















