"""
Refactored Registry System

Sistema de registro y dependencias refactorizado para el AI History Comparison System.
Maneja inyección de dependencias, ciclo de vida, lazy loading y optimización automática.
"""

import asyncio
import logging
import inspect
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import functools
import gc

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LifecycleStage(Enum):
    """Component lifecycle stage enumeration"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"


class DependencyType(Enum):
    """Dependency type enumeration"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    LAZY = "lazy"
    FACTORY = "factory"


class DependencyScope(Enum):
    """Dependency scope enumeration"""
    GLOBAL = "global"
    REQUEST = "request"
    SESSION = "session"
    THREAD = "thread"
    ASYNC = "async"


@dataclass
class DependencyMetadata:
    """Dependency metadata"""
    name: str
    dependency_type: DependencyType
    scope: DependencyScope
    interface: Optional[Type] = None
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    lifecycle_hooks: Dict[str, Callable] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentInstance:
    """Component instance with lifecycle management"""
    instance: Any
    metadata: DependencyMetadata
    lifecycle_stage: LifecycleStage = LifecycleStage.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    is_disposed: bool = False


class LifecycleManager:
    """Component lifecycle manager"""
    
    def __init__(self):
        self._hooks: Dict[LifecycleStage, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    def add_hook(self, stage: LifecycleStage, hook: Callable) -> None:
        """Add lifecycle hook"""
        if stage not in self._hooks:
            self._hooks[stage] = []
        self._hooks[stage].append(hook)
    
    def remove_hook(self, stage: LifecycleStage, hook: Callable) -> None:
        """Remove lifecycle hook"""
        if stage in self._hooks and hook in self._hooks[stage]:
            self._hooks[stage].remove(hook)
    
    async def execute_hooks(self, stage: LifecycleStage, instance: Any) -> None:
        """Execute lifecycle hooks for stage"""
        if stage not in self._hooks:
            return
        
        for hook in self._hooks[stage]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(instance)
                else:
                    hook(instance)
            except Exception as e:
                logger.error(f"Error in lifecycle hook for {stage.value}: {e}")


class DependencyResolver:
    """Dependency resolver with cycle detection"""
    
    def __init__(self):
        self._resolving: Set[str] = set()
        self._resolved: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def resolve_dependencies(self, name: str, registry: 'RefactoredRegistry') -> List[Any]:
        """Resolve dependencies with cycle detection"""
        async with self._lock:
            if name in self._resolving:
                raise ValueError(f"Circular dependency detected: {name}")
            
            if name in self._resolved:
                return []
            
            self._resolving.add(name)
            
            try:
                metadata = registry.get_metadata(name)
                if not metadata:
                    return []
                
                dependencies = []
                for dep_name in metadata.dependencies:
                    dep_instance = await registry.get(dep_name)
                    if dep_instance:
                        dependencies.append(dep_instance)
                
                self._resolved.add(name)
                return dependencies
            
            finally:
                self._resolving.discard(name)


class ScopeManager:
    """Dependency scope manager"""
    
    def __init__(self):
        self._scopes: Dict[DependencyScope, Dict[str, Any]] = {
            scope: {} for scope in DependencyScope
        }
        self._lock = asyncio.Lock()
    
    async def get_scope(self, scope: DependencyScope) -> Dict[str, Any]:
        """Get scope instances"""
        async with self._lock:
            return self._scopes[scope].copy()
    
    async def set_scope_instance(self, scope: DependencyScope, name: str, instance: Any) -> None:
        """Set scope instance"""
        async with self._lock:
            self._scopes[scope][name] = instance
    
    async def clear_scope(self, scope: DependencyScope) -> None:
        """Clear scope instances"""
        async with self._lock:
            self._scopes[scope].clear()
    
    async def cleanup_expired_scopes(self) -> None:
        """Cleanup expired scope instances"""
        async with self._lock:
            # Cleanup logic based on scope type
            for scope, instances in self._scopes.items():
                if scope == DependencyScope.REQUEST:
                    # Cleanup old request scopes
                    expired_keys = []
                    for key, instance in instances.items():
                        if hasattr(instance, 'created_at'):
                            if datetime.utcnow() - instance.created_at > timedelta(hours=1):
                                expired_keys.append(key)
                    
                    for key in expired_keys:
                        del instances[key]


class RefactoredRegistry:
    """Refactored dependency registry with advanced features"""
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyMetadata] = {}
        self._instances: Dict[str, ComponentInstance] = {}
        self._factories: Dict[str, Callable] = {}
        self._lifecycle_manager = LifecycleManager()
        self._dependency_resolver = DependencyResolver()
        self._scope_manager = ScopeManager()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval: float = 300.0  # 5 minutes
        self._max_idle_time: float = 3600.0  # 1 hour
    
    async def initialize(self) -> None:
        """Initialize registry"""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Refactored registry initialized")
    
    async def register_singleton(self, name: str, interface: Type, implementation: Type = None,
                                dependencies: List[str] = None, configuration: Dict[str, Any] = None) -> None:
        """Register singleton dependency"""
        await self._register_dependency(
            name=name,
            dependency_type=DependencyType.SINGLETON,
            scope=DependencyScope.GLOBAL,
            interface=interface,
            implementation=implementation,
            dependencies=dependencies or [],
            configuration=configuration or {}
        )
    
    async def register_transient(self, name: str, interface: Type, implementation: Type = None,
                                dependencies: List[str] = None, configuration: Dict[str, Any] = None) -> None:
        """Register transient dependency"""
        await self._register_dependency(
            name=name,
            dependency_type=DependencyType.TRANSIENT,
            scope=DependencyScope.GLOBAL,
            interface=interface,
            implementation=implementation,
            dependencies=dependencies or [],
            configuration=configuration or {}
        )
    
    async def register_scoped(self, name: str, interface: Type, implementation: Type = None,
                             scope: DependencyScope = DependencyScope.REQUEST,
                             dependencies: List[str] = None, configuration: Dict[str, Any] = None) -> None:
        """Register scoped dependency"""
        await self._register_dependency(
            name=name,
            dependency_type=DependencyType.SCOPED,
            scope=scope,
            interface=interface,
            implementation=implementation,
            dependencies=dependencies or [],
            configuration=configuration or {}
        )
    
    async def register_factory(self, name: str, factory: Callable, interface: Type = None,
                              dependencies: List[str] = None, configuration: Dict[str, Any] = None) -> None:
        """Register factory dependency"""
        await self._register_dependency(
            name=name,
            dependency_type=DependencyType.FACTORY,
            scope=DependencyScope.GLOBAL,
            interface=interface,
            factory=factory,
            dependencies=dependencies or [],
            configuration=configuration or {}
        )
    
    async def _register_dependency(self, name: str, dependency_type: DependencyType,
                                  scope: DependencyScope, interface: Type = None,
                                  implementation: Type = None, factory: Callable = None,
                                  dependencies: List[str] = None, configuration: Dict[str, Any] = None) -> None:
        """Register dependency with metadata"""
        async with self._lock:
            metadata = DependencyMetadata(
                name=name,
                dependency_type=dependency_type,
                scope=scope,
                interface=interface,
                implementation=implementation,
                factory=factory,
                dependencies=dependencies or [],
                configuration=configuration or {}
            )
            
            self._dependencies[name] = metadata
            
            if factory:
                self._factories[name] = factory
            
            logger.info(f"Registered dependency: {name} ({dependency_type.value})")
    
    async def get(self, name: str, scope: DependencyScope = None) -> Any:
        """Get dependency instance"""
        async with self._lock:
            metadata = self._dependencies.get(name)
            if not metadata:
                raise ValueError(f"Dependency not registered: {name}")
            
            # Check scope-specific instances first
            if scope and metadata.scope == scope:
                scope_instances = await self._scope_manager.get_scope(scope)
                if name in scope_instances:
                    return scope_instances[name]
            
            # Check existing instances
            if name in self._instances:
                instance = self._instances[name]
                instance.last_accessed = datetime.utcnow()
                instance.access_count += 1
                return instance.instance
            
            # Create new instance
            instance = await self._create_instance(name, metadata, scope)
            return instance
    
    async def _create_instance(self, name: str, metadata: DependencyMetadata, scope: DependencyScope = None) -> Any:
        """Create dependency instance"""
        try:
            # Resolve dependencies
            dependencies = await self._dependency_resolver.resolve_dependencies(name, self)
            
            # Create instance
            if metadata.factory:
                instance = await self._create_from_factory(metadata, dependencies)
            else:
                instance = await self._create_from_class(metadata, dependencies)
            
            # Initialize instance
            await self._lifecycle_manager.execute_hooks(LifecycleStage.INITIALIZING, instance)
            
            if hasattr(instance, 'initialize'):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    instance.initialize()
            
            await self._lifecycle_manager.execute_hooks(LifecycleStage.INITIALIZED, instance)
            
            # Store instance based on type
            component_instance = ComponentInstance(
                instance=instance,
                metadata=metadata,
                lifecycle_stage=LifecycleStage.INITIALIZED
            )
            
            if metadata.dependency_type == DependencyType.SINGLETON:
                self._instances[name] = component_instance
            elif metadata.dependency_type == DependencyType.SCOPED and scope:
                await self._scope_manager.set_scope_instance(scope, name, instance)
            
            logger.info(f"Created instance: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance {name}: {e}")
            raise
    
    async def _create_from_factory(self, metadata: DependencyMetadata, dependencies: List[Any]) -> Any:
        """Create instance from factory"""
        factory = self._factories.get(metadata.name)
        if not factory:
            raise ValueError(f"Factory not found: {metadata.name}")
        
        # Inject dependencies into factory
        if asyncio.iscoroutinefunction(factory):
            return await factory(*dependencies)
        else:
            return factory(*dependencies)
    
    async def _create_from_class(self, metadata: DependencyMetadata, dependencies: List[Any]) -> Any:
        """Create instance from class"""
        implementation = metadata.implementation
        if not implementation:
            raise ValueError(f"Implementation not specified: {metadata.name}")
        
        # Get constructor parameters
        signature = inspect.signature(implementation.__init__)
        params = list(signature.parameters.keys())[1:]  # Skip 'self'
        
        # Match dependencies to parameters
        kwargs = {}
        for i, param in enumerate(params):
            if i < len(dependencies):
                kwargs[param] = dependencies[i]
        
        return implementation(**kwargs)
    
    async def start(self, name: str) -> None:
        """Start dependency instance"""
        instance = await self.get(name)
        
        await self._lifecycle_manager.execute_hooks(LifecycleStage.STARTING, instance)
        
        if hasattr(instance, 'start'):
            if asyncio.iscoroutinefunction(instance.start):
                await instance.start()
            else:
                instance.start()
        
        await self._lifecycle_manager.execute_hooks(LifecycleStage.STARTED, instance)
        
        # Update lifecycle stage
        if name in self._instances:
            self._instances[name].lifecycle_stage = LifecycleStage.STARTED
    
    async def stop(self, name: str) -> None:
        """Stop dependency instance"""
        if name not in self._instances:
            return
        
        instance = self._instances[name].instance
        
        await self._lifecycle_manager.execute_hooks(LifecycleStage.STOPPING, instance)
        
        if hasattr(instance, 'stop'):
            if asyncio.iscoroutinefunction(instance.stop):
                await instance.stop()
            else:
                instance.stop()
        
        await self._lifecycle_manager.execute_hooks(LifecycleStage.STOPPED, instance)
        
        # Update lifecycle stage
        self._instances[name].lifecycle_stage = LifecycleStage.STOPPED
    
    async def dispose(self, name: str) -> None:
        """Dispose dependency instance"""
        if name not in self._instances:
            return
        
        instance = self._instances[name].instance
        
        await self._lifecycle_manager.execute_hooks(LifecycleStage.DESTROYED, instance)
        
        if hasattr(instance, 'dispose'):
            if asyncio.iscoroutinefunction(instance.dispose):
                await instance.dispose()
            else:
                instance.dispose()
        
        # Mark as disposed
        self._instances[name].is_disposed = True
        self._instances[name].lifecycle_stage = LifecycleStage.DESTROYED
        
        # Remove from instances if transient
        metadata = self._dependencies.get(name)
        if metadata and metadata.dependency_type == DependencyType.TRANSIENT:
            del self._instances[name]
    
    def get_metadata(self, name: str) -> Optional[DependencyMetadata]:
        """Get dependency metadata"""
        return self._dependencies.get(name)
    
    def list_dependencies(self) -> List[str]:
        """List all registered dependencies"""
        return list(self._dependencies.keys())
    
    def list_instances(self) -> List[str]:
        """List all created instances"""
        return list(self._instances.keys())
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired instances"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_instances()
                await self._scope_manager.cleanup_expired_scopes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_instances(self) -> None:
        """Cleanup expired instances"""
        async with self._lock:
            expired_instances = []
            current_time = datetime.utcnow()
            
            for name, instance in self._instances.items():
                if instance.is_disposed:
                    continue
                
                # Check if instance is idle for too long
                idle_time = current_time - instance.last_accessed
                if idle_time.total_seconds() > self._max_idle_time:
                    expired_instances.append(name)
            
            # Dispose expired instances
            for name in expired_instances:
                await self.dispose(name)
                logger.info(f"Disposed expired instance: {name}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get registry health status"""
        return {
            "dependencies_count": len(self._dependencies),
            "instances_count": len(self._instances),
            "factories_count": len(self._factories),
            "cleanup_interval": self._cleanup_interval,
            "max_idle_time": self._max_idle_time,
            "instances_by_stage": {
                stage.value: len([
                    inst for inst in self._instances.values()
                    if inst.lifecycle_stage == stage
                ])
                for stage in LifecycleStage
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown registry"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Stop all instances
        for name in list(self._instances.keys()):
            try:
                await self.stop(name)
                await self.dispose(name)
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        logger.info("Refactored registry shutdown")


# Global registry
registry = RefactoredRegistry()


# Convenience functions
async def register_singleton(name: str, interface: Type, implementation: Type = None, **kwargs):
    """Register singleton dependency"""
    await registry.register_singleton(name, interface, implementation, **kwargs)


async def register_transient(name: str, interface: Type, implementation: Type = None, **kwargs):
    """Register transient dependency"""
    await registry.register_transient(name, interface, implementation, **kwargs)


async def register_scoped(name: str, interface: Type, implementation: Type = None, **kwargs):
    """Register scoped dependency"""
    await registry.register_scoped(name, interface, implementation, **kwargs)


async def register_factory(name: str, factory: Callable, interface: Type = None, **kwargs):
    """Register factory dependency"""
    await registry.register_factory(name, factory, interface, **kwargs)


async def get_dependency(name: str, scope: DependencyScope = None):
    """Get dependency instance"""
    return await registry.get(name, scope)


async def start_dependency(name: str):
    """Start dependency instance"""
    await registry.start(name)


async def stop_dependency(name: str):
    """Stop dependency instance"""
    await registry.stop(name)


# Dependency injection decorators
def inject(name: str, scope: DependencyScope = None):
    """Dependency injection decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            dependency = await get_dependency(name, scope)
            return await func(dependency, *args, **kwargs)
        return wrapper
    return decorator


def singleton(cls):
    """Singleton decorator"""
    instances = {}
    
    @functools.wraps(cls)
    async def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper


def scoped(scope: DependencyScope = DependencyScope.REQUEST):
    """Scoped decorator"""
    def decorator(cls):
        @functools.wraps(cls)
        async def wrapper(*args, **kwargs):
            # Implementation would depend on scope management
            return cls(*args, **kwargs)
        return wrapper
    return decorator





















