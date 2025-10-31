"""
Base Component Architecture for Export IA
=========================================

Modular base classes and interfaces for creating reusable, pluggable
components in the Export IA ecosystem.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type, TypeVar, Generic
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import inspect
from pathlib import Path
import importlib
import pkgutil
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')
ComponentType = TypeVar('ComponentType', bound='BaseComponent')

class ComponentStatus(Enum):
    """Component lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class ComponentPriority(Enum):
    """Component execution priority."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class ComponentMetadata:
    """Component metadata and configuration."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComponentConfig:
    """Component configuration."""
    enabled: bool = True
    priority: ComponentPriority = ComponentPriority.NORMAL
    timeout: int = 300
    retry_count: int = 3
    max_instances: int = 1
    auto_start: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentEvent:
    """Component event data."""
    event_type: str
    component_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class ComponentRegistry:
    """Registry for managing component instances and metadata."""
    
    def __init__(self):
        self._components: Dict[str, 'BaseComponent'] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._configs: Dict[str, ComponentConfig] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def register_component(
        self,
        component: 'BaseComponent',
        metadata: ComponentMetadata,
        config: ComponentConfig
    ) -> str:
        """Register a component in the registry."""
        with self._lock:
            component_id = component.id
            self._components[component_id] = component
            self._metadata[component_id] = metadata
            self._configs[component_id] = config
            self._dependencies[component_id] = metadata.dependencies.copy()
            
            logger.info(f"Registered component: {metadata.name} ({component_id})")
            return component_id
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the registry."""
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                del self._metadata[component_id]
                del self._configs[component_id]
                del self._dependencies[component_id]
                logger.info(f"Unregistered component: {component_id}")
                return True
            return False
    
    def get_component(self, component_id: str) -> Optional['BaseComponent']:
        """Get a component by ID."""
        return self._components.get(component_id)
    
    def get_metadata(self, component_id: str) -> Optional[ComponentMetadata]:
        """Get component metadata by ID."""
        return self._metadata.get(component_id)
    
    def get_config(self, component_id: str) -> Optional[ComponentConfig]:
        """Get component configuration by ID."""
        return self._configs.get(component_id)
    
    def list_components(self, tags: Optional[List[str]] = None) -> List[str]:
        """List component IDs, optionally filtered by tags."""
        component_ids = []
        
        for comp_id, metadata in self._metadata.items():
            if tags is None or any(tag in metadata.tags for tag in tags):
                component_ids.append(comp_id)
        
        return component_ids
    
    def get_dependencies(self, component_id: str) -> List[str]:
        """Get component dependencies."""
        return self._dependencies.get(component_id, [])
    
    def resolve_dependencies(self, component_id: str) -> List[str]:
        """Resolve component dependency chain."""
        resolved = []
        to_resolve = [component_id]
        
        while to_resolve:
            current = to_resolve.pop(0)
            if current not in resolved:
                resolved.append(current)
                deps = self._dependencies.get(current, [])
                to_resolve.extend(deps)
        
        return resolved

class EventBus:
    """Event bus for component communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[callable]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, callback: callable) -> None:
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: callable) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    async def publish(self, event: ComponentEvent) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            callbacks = self._subscribers.get(event.event_type, []).copy()
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}
        self._lock = threading.RLock()
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton service."""
        with self._lock:
            self._singletons[service_type] = instance
    
    def register_factory(self, service_type: Type[T], factory: callable) -> None:
        """Register a factory for a service."""
        with self._lock:
            self._factories[service_type] = factory
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T]) -> None:
        """Register a transient service."""
        with self._lock:
            self._services[service_type] = implementation_type
    
    def get(self, service_type: Type[T]) -> T:
        """Get a service instance."""
        with self._lock:
            # Check singletons first
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            # Check factories
            if service_type in self._factories:
                return self._factories[service_type]()
            
            # Check transient services
            if service_type in self._services:
                impl_type = self._services[service_type]
                return impl_type()
            
            raise ValueError(f"Service {service_type} not registered")

class BaseComponent(ABC):
    """Base class for all Export IA components."""
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "Export IA",
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.tags = tags or []
        
        self.status = ComponentStatus.UNINITIALIZED
        self.config: Dict[str, Any] = {}
        self.metadata: Optional[ComponentMetadata] = None
        
        # Component lifecycle
        self._initialized = False
        self._running = False
        self._lock = threading.RLock()
        
        # Dependencies
        self._registry: Optional[ComponentRegistry] = None
        self._event_bus: Optional[EventBus] = None
        self._container: Optional[DependencyContainer] = None
        
        # Execution
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}-")
        self._tasks: List[asyncio.Task] = []
        
        logger.info(f"Created component: {name} ({self.id})")
    
    @property
    def is_ready(self) -> bool:
        """Check if component is ready for use."""
        return self.status == ComponentStatus.READY
    
    @property
    def is_running(self) -> bool:
        """Check if component is running."""
        return self.status == ComponentStatus.RUNNING
    
    def set_registry(self, registry: ComponentRegistry) -> None:
        """Set the component registry."""
        self._registry = registry
    
    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus."""
        self._event_bus = event_bus
    
    def set_container(self, container: DependencyContainer) -> None:
        """Set the dependency container."""
        self._container = container
    
    async def initialize(self, config: Dict[str, Any] = None) -> None:
        """Initialize the component."""
        with self._lock:
            if self._initialized:
                return
            
            self.status = ComponentStatus.INITIALIZING
            self.config = config or {}
            
            try:
                await self._on_initialize()
                self._initialized = True
                self.status = ComponentStatus.READY
                
                if self._event_bus:
                    await self._event_bus.publish(ComponentEvent(
                        event_type="component.initialized",
                        component_id=self.id,
                        data={"name": self.name, "version": self.version}
                    ))
                
                logger.info(f"Initialized component: {self.name}")
                
            except Exception as e:
                self.status = ComponentStatus.ERROR
                logger.error(f"Failed to initialize component {self.name}: {e}")
                raise
    
    async def start(self) -> None:
        """Start the component."""
        with self._lock:
            if not self._initialized:
                await self.initialize()
            
            if self._running:
                return
            
            self.status = ComponentStatus.RUNNING
            self._running = True
            
            try:
                await self._on_start()
                
                if self._event_bus:
                    await self._event_bus.publish(ComponentEvent(
                        event_type="component.started",
                        component_id=self.id,
                        data={"name": self.name}
                    ))
                
                logger.info(f"Started component: {self.name}")
                
            except Exception as e:
                self.status = ComponentStatus.ERROR
                self._running = False
                logger.error(f"Failed to start component {self.name}: {e}")
                raise
    
    async def stop(self) -> None:
        """Stop the component."""
        with self._lock:
            if not self._running:
                return
            
            self.status = ComponentStatus.STOPPING
            self._running = False
            
            try:
                # Cancel all running tasks
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete
                if self._tasks:
                    await asyncio.gather(*self._tasks, return_exceptions=True)
                
                await self._on_stop()
                self.status = ComponentStatus.STOPPED
                
                if self._event_bus:
                    await self._event_bus.publish(ComponentEvent(
                        event_type="component.stopped",
                        component_id=self.id,
                        data={"name": self.name}
                    ))
                
                logger.info(f"Stopped component: {self.name}")
                
            except Exception as e:
                self.status = ComponentStatus.ERROR
                logger.error(f"Error stopping component {self.name}: {e}")
                raise
    
    async def pause(self) -> None:
        """Pause the component."""
        with self._lock:
            if self.status != ComponentStatus.RUNNING:
                return
            
            self.status = ComponentStatus.PAUSED
            await self._on_pause()
            
            if self._event_bus:
                await self._event_bus.publish(ComponentEvent(
                    event_type="component.paused",
                    component_id=self.id,
                    data={"name": self.name}
                ))
            
            logger.info(f"Paused component: {self.name}")
    
    async def resume(self) -> None:
        """Resume the component."""
        with self._lock:
            if self.status != ComponentStatus.PAUSED:
                return
            
            self.status = ComponentStatus.RUNNING
            await self._on_resume()
            
            if self._event_bus:
                await self._event_bus.publish(ComponentEvent(
                    event_type="component.resumed",
                    component_id=self.id,
                    data={"name": self.name}
                ))
            
            logger.info(f"Resumed component: {self.name}")
    
    def get_dependency(self, service_type: Type[T]) -> T:
        """Get a dependency from the container."""
        if not self._container:
            raise RuntimeError("Dependency container not set")
        return self._container.get(service_type)
    
    def get_component(self, component_id: str) -> Optional['BaseComponent']:
        """Get a component from the registry."""
        if not self._registry:
            return None
        return self._registry.get_component(component_id)
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to the event bus."""
        if not self._event_bus:
            return
        
        event = ComponentEvent(
            event_type=event_type,
            component_id=self.id,
            data=data
        )
        await self._event_bus.publish(event)
    
    def create_task(self, coro) -> asyncio.Task:
        """Create and track an async task."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        
        # Remove completed tasks
        self._tasks = [t for t in self._tasks if not t.done()]
        
        return task
    
    def get_metadata(self) -> ComponentMetadata:
        """Get component metadata."""
        if not self.metadata:
            self.metadata = ComponentMetadata(
                name=self.name,
                version=self.version,
                description=self.description,
                author=self.author,
                dependencies=self.dependencies,
                tags=self.tags
            )
        return self.metadata
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "initialized": self._initialized,
            "running": self._running,
            "active_tasks": len([t for t in self._tasks if not t.done()]),
            "config": self.config,
            "dependencies": self.dependencies
        }
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _on_initialize(self) -> None:
        """Called during component initialization."""
        pass
    
    @abstractmethod
    async def _on_start(self) -> None:
        """Called when component starts."""
        pass
    
    @abstractmethod
    async def _on_stop(self) -> None:
        """Called when component stops."""
        pass
    
    async def _on_pause(self) -> None:
        """Called when component is paused."""
        pass
    
    async def _on_resume(self) -> None:
        """Called when component is resumed."""
        pass

class ServiceComponent(BaseComponent):
    """Base class for service components."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tags.append("service")
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data through the service."""
        pass

class ProcessorComponent(ServiceComponent):
    """Base class for data processing components."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tags.append("processor")
    
    @abstractmethod
    async def process_batch(self, data_list: List[Any]) -> List[Any]:
        """Process a batch of data."""
        pass

class AnalyzerComponent(ServiceComponent):
    """Base class for analysis components."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tags.append("analyzer")
    
    @abstractmethod
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data and return results."""
        pass

class TransformerComponent(ServiceComponent):
    """Base class for data transformation components."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tags.append("transformer")
    
    @abstractmethod
    async def transform(self, data: Any) -> Any:
        """Transform data."""
        pass

class ValidatorComponent(ServiceComponent):
    """Base class for validation components."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tags.append("validator")
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """Validate data and return result."""
        pass

# Global instances
_global_registry: Optional[ComponentRegistry] = None
_global_event_bus: Optional[EventBus] = None
_global_container: Optional[DependencyContainer] = None

def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry

def get_global_event_bus() -> EventBus:
    """Get the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus

def get_global_container() -> DependencyContainer:
    """Get the global dependency container."""
    global _global_container
    if _global_container is None:
        _global_container = DependencyContainer()
    return _global_container



























