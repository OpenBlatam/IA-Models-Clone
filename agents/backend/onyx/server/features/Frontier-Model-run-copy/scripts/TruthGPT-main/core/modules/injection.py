"""
Dependency Injection System
Advanced dependency injection for modular architecture
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Type, Callable, TypeVar, Generic, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Scope(Enum):
    """Dependency injection scopes"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

class ServiceLifetime(Enum):
    """Service lifetime"""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    SESSION = "session"

@dataclass
class ServiceDescriptor:
    """Service descriptor for dependency injection"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: Scope = Scope.SINGLETON
    lifetime: ServiceLifetime = ServiceLifetime.PERMANENT
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class InjectionContext:
    """Injection context for scoped services"""
    context_id: str
    services: Dict[Type, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class ServiceContainer:
    """Container for dependency injection"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.Lock()
        self._context_counter = 0
    
    def register_singleton(self, 
                          service_type: Type, 
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None,
                          instance: Optional[Any] = None) -> 'ServiceContainer':
        """Register singleton service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            scope=Scope.SINGLETON
        )
        
        with self._lock:
            self._services[service_type] = descriptor
        
        logger.info(f"Registered singleton: {service_type.__name__}")
        return self
    
    def register_transient(self, 
                          service_type: Type, 
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None) -> 'ServiceContainer':
        """Register transient service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            scope=Scope.TRANSIENT
        )
        
        with self._lock:
            self._services[service_type] = descriptor
        
        logger.info(f"Registered transient: {service_type.__name__}")
        return self
    
    def register_scoped(self, 
                     service_type: Type, 
                     implementation_type: Optional[Type] = None,
                     factory: Optional[Callable] = None) -> 'ServiceContainer':
        """Register scoped service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            scope=Scope.SCOPED
        )
        
        with self._lock:
            self._services[service_type] = descriptor
        
        logger.info(f"Registered scoped: {service_type.__name__}")
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'ServiceContainer':
        """Register service instance"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            scope=Scope.SINGLETON
        )
        
        with self._lock:
            self._services[service_type] = descriptor
            self._instances[service_type] = instance
        
        logger.info(f"Registered instance: {service_type.__name__}")
        return self
    
    def get(self, service_type: Type, context_id: Optional[str] = None) -> Optional[Any]:
        """Get service instance"""
        with self._lock:
            if service_type not in self._services:
                logger.error(f"Service not registered: {service_type.__name__}")
                return None
            
            descriptor = self._services[service_type]
            
            # Handle different scopes
            if descriptor.scope == Scope.SINGLETON:
                if service_type not in self._instances:
                    instance = self._create_instance(descriptor, context_id)
                    if instance:
                        self._instances[service_type] = instance
                return self._instances.get(service_type)
            
            elif descriptor.scope == Scope.TRANSIENT:
                return self._create_instance(descriptor, context_id)
            
            elif descriptor.scope == Scope.SCOPED:
                if not context_id:
                    logger.error(f"Context ID required for scoped service: {service_type.__name__}")
                    return None
                
                if context_id not in self._scoped_instances:
                    self._scoped_instances[context_id] = {}
                
                if service_type not in self._scoped_instances[context_id]:
                    instance = self._create_instance(descriptor, context_id)
                    if instance:
                        self._scoped_instances[context_id][service_type] = instance
                
                return self._scoped_instances[context_id].get(service_type)
            
            return None
    
    def _create_instance(self, descriptor: ServiceDescriptor, context_id: Optional[str] = None) -> Optional[Any]:
        """Create service instance"""
        try:
            # Use existing instance if available
            if descriptor.instance:
                return descriptor.instance
            
            # Use factory if available
            if descriptor.factory:
                return descriptor.factory()
            
            # Use implementation type
            implementation_type = descriptor.implementation_type or descriptor.service_type
            
            # Get constructor parameters
            constructor_params = self._get_constructor_params(implementation_type)
            
            # Resolve dependencies
            resolved_params = {}
            for param_name, param_type in constructor_params.items():
                if param_type in self._services:
                    dependency = self.get(param_type, context_id)
                    if dependency:
                        resolved_params[param_name] = dependency
                    else:
                        logger.error(f"Failed to resolve dependency: {param_type.__name__}")
                        return None
                else:
                    logger.warning(f"Dependency not registered: {param_type.__name__}")
            
            # Create instance
            instance = implementation_type(**resolved_params)
            logger.info(f"Created instance: {implementation_type.__name__}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance of {descriptor.service_type.__name__}: {e}")
            return None
    
    def _get_constructor_params(self, service_type: Type) -> Dict[str, Type]:
        """Get constructor parameters for a service type"""
        try:
            signature = inspect.signature(service_type.__init__)
            params = {}
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    params[param_name] = param.annotation
                else:
                    # Try to infer type from default value
                    if param.default != inspect.Parameter.empty:
                        params[param_name] = type(param.default)
            
            return params
        except Exception as e:
            logger.error(f"Failed to get constructor params for {service_type.__name__}: {e}")
            return {}
    
    def create_scope(self) -> str:
        """Create a new scope"""
        with self._lock:
            self._context_counter += 1
            context_id = f"scope_{self._context_counter}"
            self._scoped_instances[context_id] = {}
            return context_id
    
    def dispose_scope(self, context_id: str) -> None:
        """Dispose a scope"""
        with self._lock:
            if context_id in self._scoped_instances:
                # Call dispose on disposable services
                for service_type, instance in self._scoped_instances[context_id].items():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception as e:
                            logger.error(f"Error disposing {service_type.__name__}: {e}")
                
                del self._scoped_instances[context_id]
                logger.info(f"Disposed scope: {context_id}")
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        with self._lock:
            return service_type in self._services
    
    def list_services(self) -> List[Type]:
        """List registered services"""
        with self._lock:
            return list(self._services.keys())
    
    def get_service_info(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """Get service information"""
        with self._lock:
            return self._services.get(service_type)

class DependencyInjector:
    """Main dependency injector"""
    
    def __init__(self):
        self.container = ServiceContainer()
        self._contexts: Dict[str, InjectionContext] = {}
        self._lock = threading.Lock()
    
    def register_singleton(self, service_type: Type, **kwargs) -> 'DependencyInjector':
        """Register singleton service"""
        self.container.register_singleton(service_type, **kwargs)
        return self
    
    def register_transient(self, service_type: Type, **kwargs) -> 'DependencyInjector':
        """Register transient service"""
        self.container.register_transient(service_type, **kwargs)
        return self
    
    def register_scoped(self, service_type: Type, **kwargs) -> 'DependencyInjector':
        """Register scoped service"""
        self.container.register_scoped(service_type, **kwargs)
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'DependencyInjector':
        """Register service instance"""
        self.container.register_instance(service_type, instance)
        return self
    
    def get(self, service_type: Type, context_id: Optional[str] = None) -> Optional[Any]:
        """Get service instance"""
        return self.container.get(service_type, context_id)
    
    def create_scope(self) -> str:
        """Create a new scope"""
        return self.container.create_scope()
    
    def dispose_scope(self, context_id: str) -> None:
        """Dispose a scope"""
        self.container.dispose_scope(context_id)
    
    def create_context(self, context_id: Optional[str] = None) -> str:
        """Create injection context"""
        if not context_id:
            context_id = f"context_{int(time.time() * 1000)}"
        
        with self._lock:
            self._contexts[context_id] = InjectionContext(context_id=context_id)
        
        return context_id
    
    def dispose_context(self, context_id: str) -> None:
        """Dispose injection context"""
        with self._lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
        
        self.dispose_scope(context_id)
    
    def get_context(self, context_id: str) -> Optional[InjectionContext]:
        """Get injection context"""
        with self._lock:
            return self._contexts.get(context_id)
    
    def list_contexts(self) -> List[str]:
        """List active contexts"""
        with self._lock:
            return list(self._contexts.keys())

class AutoInjector:
    """Automatic dependency injection"""
    
    def __init__(self, injector: DependencyInjector):
        self.injector = injector
    
    def inject(self, obj: Any, context_id: Optional[str] = None) -> Any:
        """Inject dependencies into object"""
        try:
            # Get object's class
            obj_class = obj.__class__
            
            # Get all attributes
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                
                attr = getattr(obj, attr_name)
                
                # Check if attribute has injection annotation
                if hasattr(attr, '__annotations__'):
                    for param_name, param_type in attr.__annotations__.items():
                        if self.injector.container.is_registered(param_type):
                            dependency = self.injector.get(param_type, context_id)
                            if dependency:
                                setattr(obj, param_name, dependency)
            
            return obj
        except Exception as e:
            logger.error(f"Failed to inject dependencies: {e}")
            return obj
    
    def create_with_injection(self, 
                             class_type: Type, 
                             context_id: Optional[str] = None,
                             **kwargs) -> Optional[Any]:
        """Create instance with dependency injection"""
        try:
            # Get constructor parameters
            signature = inspect.signature(class_type.__init__)
            injected_params = {}
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param_name in kwargs:
                    injected_params[param_name] = kwargs[param_name]
                elif param.annotation != inspect.Parameter.empty:
                    if self.injector.container.is_registered(param.annotation):
                        dependency = self.injector.get(param.annotation, context_id)
                        if dependency:
                            injected_params[param_name] = dependency
                        else:
                            logger.warning(f"Failed to resolve dependency: {param.annotation.__name__}")
            
            # Create instance
            instance = class_type(**injected_params)
            
            # Inject additional dependencies
            return self.inject(instance, context_id)
            
        except Exception as e:
            logger.error(f"Failed to create instance with injection: {e}")
            return None

class ServiceLocator:
    """Service locator pattern implementation"""
    
    def __init__(self, injector: DependencyInjector):
        self.injector = injector
        self._cache: Dict[Type, Any] = {}
        self._lock = threading.Lock()
    
    def get(self, service_type: Type, context_id: Optional[str] = None) -> Optional[Any]:
        """Get service with caching"""
        with self._lock:
            cache_key = f"{service_type.__name__}_{context_id or 'default'}"
            
            if cache_key not in self._cache:
                instance = self.injector.get(service_type, context_id)
                if instance:
                    self._cache[cache_key] = instance
            
            return self._cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear service cache"""
        with self._lock:
            self._cache.clear()
    
    def is_available(self, service_type: Type) -> bool:
        """Check if service is available"""
        return self.injector.container.is_registered(service_type)

class InjectionBuilder:
    """Builder for dependency injection setup"""
    
    def __init__(self):
        self.injector = DependencyInjector()
        self.services: List[Dict[str, Any]] = []
    
    def add_singleton(self, service_type: Type, **kwargs) -> 'InjectionBuilder':
        """Add singleton service"""
        self.services.append({
            "type": "singleton",
            "service_type": service_type,
            **kwargs
        })
        return self
    
    def add_transient(self, service_type: Type, **kwargs) -> 'InjectionBuilder':
        """Add transient service"""
        self.services.append({
            "type": "transient",
            "service_type": service_type,
            **kwargs
        })
        return self
    
    def add_scoped(self, service_type: Type, **kwargs) -> 'InjectionBuilder':
        """Add scoped service"""
        self.services.append({
            "type": "scoped",
            "service_type": service_type,
            **kwargs
        })
        return self
    
    def add_instance(self, service_type: Type, instance: Any) -> 'InjectionBuilder':
        """Add service instance"""
        self.services.append({
            "type": "instance",
            "service_type": service_type,
            "instance": instance
        })
        return self
    
    def build(self) -> DependencyInjector:
        """Build dependency injection system"""
        for service in self.services:
            service_type = service["service_type"]
            service_kwargs = {k: v for k, v in service.items() if k not in ["type", "service_type"]}
            
            if service["type"] == "singleton":
                self.injector.register_singleton(service_type, **service_kwargs)
            elif service["type"] == "transient":
                self.injector.register_transient(service_type, **service_kwargs)
            elif service["type"] == "scoped":
                self.injector.register_scoped(service_type, **service_kwargs)
            elif service["type"] == "instance":
                self.injector.register_instance(service_type, service["instance"])
        
        return self.injector

