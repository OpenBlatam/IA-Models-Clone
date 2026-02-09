"""
Dependency Injection System
==========================

Ultra-modular dependency injection with advanced patterns.
"""

import logging
import threading
from typing import Dict, Any, Optional, Type, List, Callable, Union, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools

logger = logging.getLogger(__name__)

class Scope(str, Enum):
    """Dependency scopes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    REQUEST = "request"

@dataclass
class DependencyInfo:
    """Dependency information."""
    name: str
    dependency_type: Type
    implementation: Type
    scope: Scope
    factory: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    created_at: float = field(default_factory=lambda: __import__('time').time())

class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        self._services: Dict[str, DependencyInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._current_scope: Optional[str] = None
    
    def register_singleton(self, name: str, implementation: Type, 
                          dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a singleton service."""
        try:
            with self._lock:
                self._services[name] = DependencyInfo(
                    name=name,
                    dependency_type=implementation,
                    implementation=implementation,
                    scope=Scope.SINGLETON,
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                logger.info(f"Singleton service {name} registered")
        except Exception as e:
            logger.error(f"Failed to register singleton {name}: {str(e)}")
            raise
    
    def register_transient(self, name: str, implementation: Type,
                          dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a transient service."""
        try:
            with self._lock:
                self._services[name] = DependencyInfo(
                    name=name,
                    dependency_type=implementation,
                    implementation=implementation,
                    scope=Scope.TRANSIENT,
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                logger.info(f"Transient service {name} registered")
        except Exception as e:
            logger.error(f"Failed to register transient {name}: {str(e)}")
            raise
    
    def register_scoped(self, name: str, implementation: Type,
                       dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a scoped service."""
        try:
            with self._lock:
                self._services[name] = DependencyInfo(
                    name=name,
                    dependency_type=implementation,
                    implementation=implementation,
                    scope=Scope.SCOPED,
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                logger.info(f"Scoped service {name} registered")
        except Exception as e:
            logger.error(f"Failed to register scoped {name}: {str(e)}")
            raise
    
    def register_factory(self, name: str, factory: Callable, scope: Scope = Scope.TRANSIENT,
                        dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a factory service."""
        try:
            with self._lock:
                self._services[name] = DependencyInfo(
                    name=name,
                    dependency_type=type(factory),
                    implementation=type(factory),
                    scope=scope,
                    factory=factory,
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                logger.info(f"Factory service {name} registered")
        except Exception as e:
            logger.error(f"Failed to register factory {name}: {str(e)}")
            raise
    
    def get(self, name: str) -> Any:
        """Get a service instance."""
        try:
            with self._lock:
                if name not in self._services:
                    logger.error(f"Service {name} not found")
                    return None
                
                service_info = self._services[name]
                
                if service_info.scope == Scope.SINGLETON:
                    return self._get_singleton(name, service_info)
                elif service_info.scope == Scope.TRANSIENT:
                    return self._get_transient(name, service_info)
                elif service_info.scope == Scope.SCOPED:
                    return self._get_scoped(name, service_info)
                elif service_info.scope == Scope.REQUEST:
                    return self._get_request(name, service_info)
                else:
                    logger.error(f"Unknown scope: {service_info.scope}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get service {name}: {str(e)}")
            return None
    
    def _get_singleton(self, name: str, service_info: DependencyInfo) -> Any:
        """Get singleton instance."""
        try:
            if name in self._instances:
                return self._instances[name]
            
            instance = self._create_instance(name, service_info)
            self._instances[name] = instance
            return instance
            
        except Exception as e:
            logger.error(f"Failed to get singleton {name}: {str(e)}")
            return None
    
    def _get_transient(self, name: str, service_info: DependencyInfo) -> Any:
        """Get transient instance."""
        try:
            return self._create_instance(name, service_info)
        except Exception as e:
            logger.error(f"Failed to get transient {name}: {str(e)}")
            return None
    
    def _get_scoped(self, name: str, service_info: DependencyInfo) -> Any:
        """Get scoped instance."""
        try:
            if not self._current_scope:
                logger.error("No current scope for scoped service")
                return None
            
            if self._current_scope not in self._scoped_instances:
                self._scoped_instances[self._current_scope] = {}
            
            if name in self._scoped_instances[self._current_scope]:
                return self._scoped_instances[self._current_scope][name]
            
            instance = self._create_instance(name, service_info)
            self._scoped_instances[self._current_scope][name] = instance
            return instance
            
        except Exception as e:
            logger.error(f"Failed to get scoped {name}: {str(e)}")
            return None
    
    def _get_request(self, name: str, service_info: DependencyInfo) -> Any:
        """Get request instance."""
        try:
            # For request scope, create new instance each time
            return self._create_instance(name, service_info)
        except Exception as e:
            logger.error(f"Failed to get request {name}: {str(e)}")
            return None
    
    def _create_instance(self, name: str, service_info: DependencyInfo) -> Any:
        """Create service instance."""
        try:
            # Resolve dependencies
            dependencies = {}
            for dep_name in service_info.dependencies:
                dep_instance = self.get(dep_name)
                if dep_instance is None:
                    logger.error(f"Dependency {dep_name} not found for {name}")
                    return None
                dependencies[dep_name] = dep_instance
            
            # Create instance
            if service_info.factory:
                instance = service_info.factory(**dependencies)
            else:
                # Try to inject dependencies via constructor
                if dependencies:
                    instance = service_info.implementation(**dependencies)
                else:
                    instance = service_info.implementation()
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance {name}: {str(e)}")
            return None
    
    def begin_scope(self, scope_name: str) -> None:
        """Begin a new scope."""
        try:
            with self._lock:
                self._current_scope = scope_name
                logger.info(f"Scope {scope_name} began")
        except Exception as e:
            logger.error(f"Failed to begin scope {scope_name}: {str(e)}")
            raise
    
    def end_scope(self, scope_name: str) -> None:
        """End a scope."""
        try:
            with self._lock:
                if scope_name in self._scoped_instances:
                    # Cleanup scoped instances
                    for instance in self._scoped_instances[scope_name].values():
                        if hasattr(instance, 'cleanup'):
                            instance.cleanup()
                    
                    del self._scoped_instances[scope_name]
                
                if self._current_scope == scope_name:
                    self._current_scope = None
                
                logger.info(f"Scope {scope_name} ended")
        except Exception as e:
            logger.error(f"Failed to end scope {scope_name}: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup all instances."""
        try:
            with self._lock:
                # Cleanup singleton instances
                for instance in self._instances.values():
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                
                # Cleanup scoped instances
                for scope_instances in self._scoped_instances.values():
                    for instance in scope_instances.values():
                        if hasattr(instance, 'cleanup'):
                            instance.cleanup()
                
                self._instances.clear()
                self._scoped_instances.clear()
                
                logger.info("Service container cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup service container: {str(e)}")
            raise

class DependencyInjector:
    """Dependency injector."""
    
    def __init__(self, container: ServiceContainer = None):
        self.container = container or ServiceContainer()
        self._injection_cache: Dict[str, Any] = {}
    
    def inject(self, func: Callable) -> Callable:
        """Inject dependencies into a function."""
        try:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get function signature
                sig = inspect.signature(func)
                
                # Resolve dependencies
                injected_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        continue  # Already provided
                    
                    # Try to get from container
                    if param_name in self.container._services:
                        injected_kwargs[param_name] = self.container.get(param_name)
                    elif param.annotation != inspect.Parameter.empty:
                        # Try to get by type
                        for service_name, service_info in self.container._services.items():
                            if service_info.dependency_type == param.annotation:
                                injected_kwargs[param_name] = self.container.get(service_name)
                                break
                
                # Merge with provided kwargs
                final_kwargs = {**injected_kwargs, **kwargs}
                
                return func(*args, **final_kwargs)
            
            return wrapper
        except Exception as e:
            logger.error(f"Failed to inject dependencies: {str(e)}")
            return func
    
    def inject_class(self, cls: Type) -> Type:
        """Inject dependencies into a class."""
        try:
            original_init = cls.__init__
            
            def new_init(self, *args, **kwargs):
                # Get constructor signature
                sig = inspect.signature(original_init)
                
                # Resolve dependencies
                injected_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    if param_name in kwargs:
                        continue  # Already provided
                    
                    # Try to get from container
                    if param_name in self.container._services:
                        injected_kwargs[param_name] = self.container.get(param_name)
                    elif param.annotation != inspect.Parameter.empty:
                        # Try to get by type
                        for service_name, service_info in self.container._services.items():
                            if service_info.dependency_type == param.annotation:
                                injected_kwargs[param_name] = self.container.get(service_name)
                                break
                
                # Merge with provided kwargs
                final_kwargs = {**injected_kwargs, **kwargs}
                
                return original_init(self, *args, **final_kwargs)
            
            cls.__init__ = new_init
            return cls
        except Exception as e:
            logger.error(f"Failed to inject class dependencies: {str(e)}")
            return cls

# Global service container
service_container = ServiceContainer()
dependency_injector = DependencyInjector(service_container)

# Decorators
def injectable(scope: Scope = Scope.TRANSIENT, dependencies: List[str] = None, 
              metadata: Dict[str, Any] = None):
    """Decorator to make a class injectable."""
    def decorator(cls):
        service_container.register_transient(
            cls.__name__.lower(),
            cls,
            dependencies,
            metadata
        )
        return cls
    return decorator

def singleton(scope: Scope = Scope.SINGLETON, dependencies: List[str] = None, 
              metadata: Dict[str, Any] = None):
    """Decorator to make a class a singleton."""
    def decorator(cls):
        service_container.register_singleton(
            cls.__name__.lower(),
            cls,
            dependencies,
            metadata
        )
        return cls
    return decorator

def scoped(scope: Scope = Scope.SCOPED, dependencies: List[str] = None, 
          metadata: Dict[str, Any] = None):
    """Decorator to make a class scoped."""
    def decorator(cls):
        service_container.register_scoped(
            cls.__name__.lower(),
            cls,
            dependencies,
            metadata
        )
        return cls
    return decorator

def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function."""
    return dependency_injector.inject(func)

def inject_class(cls: Type) -> Type:
    """Decorator to inject dependencies into a class."""
    return dependency_injector.inject_class(cls)









