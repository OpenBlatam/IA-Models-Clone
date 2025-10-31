"""
Component Factory Implementation

Factory pattern for creating and managing system components
with dependency injection and lifecycle management.
"""

import asyncio
import logging
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ComponentType(Enum):
    """Component type enumeration"""
    SERVICE = "service"
    ANALYZER = "analyzer"
    ENGINE = "engine"
    INTEGRATION = "integration"
    UTILITY = "utility"
    MIDDLEWARE = "middleware"
    PLUGIN = "plugin"


class LifecycleState(Enum):
    """Component lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class ComponentMetadata:
    """Component metadata"""
    name: str
    component_type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentConfig:
    """Component configuration"""
    enabled: bool = True
    auto_start: bool = True
    singleton: bool = False
    lazy_init: bool = False
    timeout: Optional[float] = None
    retry_count: int = 3
    health_check_interval: float = 30.0
    custom_config: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """Registry for component definitions and instances"""
    
    def __init__(self):
        self._definitions: Dict[str, 'ComponentDefinition'] = {}
        self._instances: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
    
    def register_definition(self, definition: 'ComponentDefinition') -> None:
        """Register a component definition"""
        self._definitions[definition.name] = definition
        self._dependencies[definition.name] = definition.dependencies.copy()
        logger.info(f"Registered component definition: {definition.name}")
    
    def get_definition(self, name: str) -> Optional['ComponentDefinition']:
        """Get component definition by name"""
        return self._definitions.get(name)
    
    def get_definitions(self, component_type: Optional[ComponentType] = None) -> List['ComponentDefinition']:
        """Get component definitions, optionally filtered by type"""
        definitions = list(self._definitions.values())
        if component_type:
            definitions = [d for d in definitions if d.component_type == component_type]
        return definitions
    
    def has_definition(self, name: str) -> bool:
        """Check if component definition exists"""
        return name in self._definitions
    
    def unregister_definition(self, name: str) -> None:
        """Unregister component definition"""
        if name in self._definitions:
            del self._definitions[name]
            if name in self._dependencies:
                del self._dependencies[name]
            logger.info(f"Unregistered component definition: {name}")
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get component dependencies"""
        return self._dependencies.get(name, [])
    
    def resolve_dependency_order(self) -> List[str]:
        """Resolve dependency order for component initialization"""
        resolved = []
        unresolved = set(self._definitions.keys())
        
        while unresolved:
            progress = False
            for name in list(unresolved):
                dependencies = self.get_dependencies(name)
                if all(dep in resolved for dep in dependencies):
                    resolved.append(name)
                    unresolved.remove(name)
                    progress = True
            
            if not progress:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected among: {unresolved}")
        
        return resolved


@dataclass
class ComponentDefinition:
    """Component definition"""
    name: str
    component_type: ComponentType
    factory: Callable[..., Any]
    metadata: ComponentMetadata
    config: ComponentConfig
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = ComponentMetadata(
                name=self.name,
                component_type=self.component_type
            )


class ComponentFactory:
    """Factory for creating and managing components"""
    
    def __init__(self):
        self._registry = ComponentRegistry()
        self._lifecycle_managers: Dict[str, 'ComponentLifecycleManager'] = {}
        self._dependency_injector = DependencyInjector()
        self._lock = asyncio.Lock()
    
    def register_component(self, name: str, component_type: ComponentType,
                          factory: Callable[..., Any], 
                          metadata: Optional[ComponentMetadata] = None,
                          config: Optional[ComponentConfig] = None,
                          dependencies: Optional[List[str]] = None) -> None:
        """Register a component definition"""
        if metadata is None:
            metadata = ComponentMetadata(name=name, component_type=component_type)
        
        if config is None:
            config = ComponentConfig()
        
        if dependencies is None:
            dependencies = []
        
        definition = ComponentDefinition(
            name=name,
            component_type=component_type,
            factory=factory,
            metadata=metadata,
            config=config,
            dependencies=dependencies
        )
        
        self._registry.register_definition(definition)
        logger.info(f"Registered component: {name} ({component_type.value})")
    
    async def create_component(self, name: str, **kwargs) -> Any:
        """Create a component instance"""
        definition = self._registry.get_definition(name)
        if not definition:
            raise ValueError(f"Component '{name}' not found")
        
        if not definition.config.enabled:
            raise ValueError(f"Component '{name}' is disabled")
        
        # Check if singleton already exists
        if definition.config.singleton and name in self._registry._singletons:
            return self._registry._singletons[name]
        
        async with self._lock:
            # Create component instance
            instance = await self._create_instance(definition, **kwargs)
            
            # Store instance
            if definition.config.singleton:
                self._registry._singletons[name] = instance
            else:
                self._registry._instances[name] = instance
            
            # Create lifecycle manager
            lifecycle_manager = ComponentLifecycleManager(instance, definition)
            self._lifecycle_managers[name] = lifecycle_manager
            
            # Initialize if not lazy
            if not definition.config.lazy_init:
                await lifecycle_manager.initialize()
            
            # Auto-start if configured
            if definition.config.auto_start and not definition.config.lazy_init:
                await lifecycle_manager.start()
            
            logger.info(f"Created component: {name}")
            return instance
    
    async def _create_instance(self, definition: ComponentDefinition, **kwargs) -> Any:
        """Create component instance with dependency injection"""
        # Resolve dependencies
        dependencies = {}
        for dep_name in definition.dependencies:
            dep_instance = await self.get_component(dep_name)
            dependencies[dep_name] = dep_instance
        
        # Merge with provided kwargs
        all_kwargs = {**dependencies, **kwargs}
        
        # Create instance
        if inspect.iscoroutinefunction(definition.factory):
            instance = await definition.factory(**all_kwargs)
        else:
            instance = definition.factory(**all_kwargs)
        
        return instance
    
    async def get_component(self, name: str) -> Any:
        """Get component instance"""
        # Check singletons first
        if name in self._registry._singletons:
            return self._registry._singletons[name]
        
        # Check regular instances
        if name in self._registry._instances:
            return self._registry._instances[name]
        
        # Create if not exists
        return await self.create_component(name)
    
    async def destroy_component(self, name: str) -> None:
        """Destroy component instance"""
        async with self._lock:
            # Stop and destroy lifecycle manager
            if name in self._lifecycle_managers:
                await self._lifecycle_managers[name].destroy()
                del self._lifecycle_managers[name]
            
            # Remove from registries
            if name in self._registry._singletons:
                del self._registry._singletons[name]
            
            if name in self._registry._instances:
                del self._registry._instances[name]
            
            logger.info(f"Destroyed component: {name}")
    
    async def start_component(self, name: str) -> None:
        """Start component"""
        if name in self._lifecycle_managers:
            await self._lifecycle_managers[name].start()
    
    async def stop_component(self, name: str) -> None:
        """Stop component"""
        if name in self._lifecycle_managers:
            await self._lifecycle_managers[name].stop()
    
    async def restart_component(self, name: str) -> None:
        """Restart component"""
        await self.stop_component(name)
        await self.start_component(name)
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get component information"""
        definition = self._registry.get_definition(name)
        if not definition:
            return None
        
        lifecycle_manager = self._lifecycle_managers.get(name)
        state = lifecycle_manager.get_state() if lifecycle_manager else LifecycleState.CREATED
        
        return {
            "name": name,
            "type": definition.component_type.value,
            "state": state.value,
            "metadata": definition.metadata.__dict__,
            "config": definition.config.__dict__,
            "dependencies": definition.dependencies
        }
    
    def list_components(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """List component names, optionally filtered by type"""
        if component_type:
            return [
                name for name, definition in self._registry._definitions.items()
                if definition.component_type == component_type
            ]
        return list(self._registry._definitions.keys())
    
    async def initialize_all_components(self) -> None:
        """Initialize all registered components in dependency order"""
        order = self._registry.resolve_dependency_order()
        
        for name in order:
            try:
                await self.create_component(name)
            except Exception as e:
                logger.error(f"Failed to initialize component '{name}': {e}")
    
    async def start_all_components(self) -> None:
        """Start all components"""
        for name in self._lifecycle_managers:
            try:
                await self.start_component(name)
            except Exception as e:
                logger.error(f"Failed to start component '{name}': {e}")
    
    async def stop_all_components(self) -> None:
        """Stop all components"""
        for name in list(self._lifecycle_managers.keys()):
            try:
                await self.stop_component(name)
            except Exception as e:
                logger.error(f"Failed to stop component '{name}': {e}")
    
    async def destroy_all_components(self) -> None:
        """Destroy all components"""
        for name in list(self._lifecycle_managers.keys()):
            try:
                await self.destroy_component(name)
            except Exception as e:
                logger.error(f"Failed to destroy component '{name}': {e}")


class ComponentLifecycleManager:
    """Manages component lifecycle"""
    
    def __init__(self, component: Any, definition: ComponentDefinition):
        self.component = component
        self.definition = definition
        self.state = LifecycleState.CREATED
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize component"""
        async with self._lock:
            if self.state != LifecycleState.CREATED:
                return
            
            self.state = LifecycleState.INITIALIZING
            try:
                if hasattr(self.component, 'initialize'):
                    if inspect.iscoroutinefunction(self.component.initialize):
                        await self.component.initialize()
                    else:
                        self.component.initialize()
                
                self.state = LifecycleState.INITIALIZED
                logger.info(f"Initialized component: {self.definition.name}")
            except Exception as e:
                self.state = LifecycleState.ERROR
                logger.error(f"Failed to initialize component '{self.definition.name}': {e}")
                raise
    
    async def start(self) -> None:
        """Start component"""
        async with self._lock:
            if self.state not in [LifecycleState.INITIALIZED, LifecycleState.STOPPED]:
                return
            
            self.state = LifecycleState.STARTING
            try:
                if hasattr(self.component, 'start'):
                    if inspect.iscoroutinefunction(self.component.start):
                        await self.component.start()
                    else:
                        self.component.start()
                
                self.state = LifecycleState.STARTED
                
                # Start health check if configured
                if self.definition.config.health_check_interval > 0:
                    self._health_check_task = asyncio.create_task(
                        self._health_check_loop()
                    )
                
                logger.info(f"Started component: {self.definition.name}")
            except Exception as e:
                self.state = LifecycleState.ERROR
                logger.error(f"Failed to start component '{self.definition.name}': {e}")
                raise
    
    async def stop(self) -> None:
        """Stop component"""
        async with self._lock:
            if self.state != LifecycleState.STARTED:
                return
            
            self.state = LifecycleState.STOPPING
            try:
                # Stop health check
                if self._health_check_task:
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass
                    self._health_check_task = None
                
                if hasattr(self.component, 'stop'):
                    if inspect.iscoroutinefunction(self.component.stop):
                        await self.component.stop()
                    else:
                        self.component.stop()
                
                self.state = LifecycleState.STOPPED
                logger.info(f"Stopped component: {self.definition.name}")
            except Exception as e:
                self.state = LifecycleState.ERROR
                logger.error(f"Failed to stop component '{self.definition.name}': {e}")
                raise
    
    async def destroy(self) -> None:
        """Destroy component"""
        async with self._lock:
            if self.state == LifecycleState.DESTROYED:
                return
            
            # Stop if running
            if self.state == LifecycleState.STARTED:
                await self.stop()
            
            try:
                if hasattr(self.component, 'destroy'):
                    if inspect.iscoroutinefunction(self.component.destroy):
                        await self.component.destroy()
                    else:
                        self.component.destroy()
                
                self.state = LifecycleState.DESTROYED
                logger.info(f"Destroyed component: {self.definition.name}")
            except Exception as e:
                self.state = LifecycleState.ERROR
                logger.error(f"Failed to destroy component '{self.definition.name}': {e}")
                raise
    
    def get_state(self) -> LifecycleState:
        """Get current component state"""
        return self.state
    
    async def _health_check_loop(self) -> None:
        """Health check loop"""
        while self.state == LifecycleState.STARTED:
            try:
                if hasattr(self.component, 'health_check'):
                    if inspect.iscoroutinefunction(self.component.health_check):
                        await self.component.health_check()
                    else:
                        self.component.health_check()
                
                await asyncio.sleep(self.definition.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for component '{self.definition.name}': {e}")
                await asyncio.sleep(self.definition.config.health_check_interval)


class DependencyInjector:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
    
    def register_service(self, service_type: Type, instance: Any) -> None:
        """Register a service instance"""
        self._services[service_type] = instance
    
    def register_factory(self, service_type: Type, factory: Callable) -> None:
        """Register a service factory"""
        self._factories[service_type] = factory
    
    def get_service(self, service_type: Type) -> Any:
        """Get service instance"""
        if service_type in self._services:
            return self._services[service_type]
        
        if service_type in self._factories:
            return self._factories[service_type]()
        
        raise ValueError(f"Service of type {service_type} not registered")
    
    def inject_dependencies(self, obj: Any) -> None:
        """Inject dependencies into object"""
        if hasattr(obj, '__annotations__'):
            for attr_name, attr_type in obj.__annotations__.items():
                if hasattr(obj, attr_name):
                    continue
                
                try:
                    service = self.get_service(attr_type)
                    setattr(obj, attr_name, service)
                except ValueError:
                    pass  # Service not registered, skip


# Global component factory instance
component_factory = ComponentFactory()


# Convenience functions
def register_component(name: str, component_type: ComponentType, factory: Callable, **kwargs):
    """Register a component"""
    component_factory.register_component(name, component_type, factory, **kwargs)


async def get_component(name: str) -> Any:
    """Get component instance"""
    return await component_factory.get_component(name)


async def create_component(name: str, **kwargs) -> Any:
    """Create component instance"""
    return await component_factory.create_component(name, **kwargs)





















