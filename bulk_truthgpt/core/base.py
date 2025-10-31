"""
Base Classes and Interfaces
===========================

Base classes and interfaces for the Bulk TruthGPT system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class ComponentStatus(str, Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ComponentInfo:
    """Component information."""
    name: str
    version: str
    status: ComponentStatus
    last_updated: datetime
    metadata: Dict[str, Any]

class BaseComponent(ABC):
    """
    Base component class for all system components.
    
    Provides common functionality for initialization, lifecycle management,
    and error handling.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.status = ComponentStatus.INITIALIZING
        self.last_updated = datetime.utcnow()
        self.metadata = {}
        self._initialized = False
        self._cleanup_tasks = []
        
    @abstractmethod
    async def _initialize_internal(self) -> None:
        """Internal initialization logic to be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _cleanup_internal(self) -> None:
        """Internal cleanup logic to be implemented by subclasses."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the component."""
        try:
            if self._initialized:
                logger.warning(f"Component {self.name} is already initialized")
                return
            
            logger.info(f"Initializing component: {self.name}")
            self.status = ComponentStatus.INITIALIZING
            
            await self._initialize_internal()
            
            self._initialized = True
            self.status = ComponentStatus.RUNNING
            self.last_updated = datetime.utcnow()
            
            logger.info(f"Component {self.name} initialized successfully")
            
        except Exception as e:
            self.status = ComponentStatus.ERROR
            logger.error(f"Failed to initialize component {self.name}: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup the component."""
        try:
            if not self._initialized:
                logger.warning(f"Component {self.name} is not initialized")
                return
            
            logger.info(f"Cleaning up component: {self.name}")
            self.status = ComponentStatus.STOPPING
            
            # Cancel all cleanup tasks
            for task in self._cleanup_tasks:
                if not task.done():
                    task.cancel()
            
            await self._cleanup_internal()
            
            self._initialized = False
            self.status = ComponentStatus.STOPPED
            self.last_updated = datetime.utcnow()
            
            logger.info(f"Component {self.name} cleaned up successfully")
            
        except Exception as e:
            self.status = ComponentStatus.ERROR
            logger.error(f"Failed to cleanup component {self.name}: {str(e)}")
            raise
    
    def get_info(self) -> ComponentInfo:
        """Get component information."""
        return ComponentInfo(
            name=self.name,
            version=self.version,
            status=self.status,
            last_updated=self.last_updated,
            metadata=self.metadata
        )
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == ComponentStatus.RUNNING and self._initialized
    
    def add_cleanup_task(self, task: asyncio.Task) -> None:
        """Add a cleanup task."""
        self._cleanup_tasks.append(task)

class BaseService(BaseComponent):
    """
    Base service class for all services.
    
    Extends BaseComponent with service-specific functionality.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.dependencies = []
        self.dependents = []
    
    def add_dependency(self, service: 'BaseService') -> None:
        """Add a service dependency."""
        if service not in self.dependencies:
            self.dependencies.append(service)
            service.add_dependent(self)
    
    def add_dependent(self, service: 'BaseService') -> None:
        """Add a dependent service."""
        if service not in self.dependents:
            self.dependents.append(service)
    
    async def _initialize_internal(self) -> None:
        """Initialize service and its dependencies."""
        # Initialize dependencies first
        for dependency in self.dependencies:
            if not dependency.is_healthy():
                await dependency.initialize()
        
        # Initialize this service
        await self._initialize_service()
    
    @abstractmethod
    async def _initialize_service(self) -> None:
        """Initialize the service (to be implemented by subclasses)."""
        pass
    
    async def _cleanup_internal(self) -> None:
        """Cleanup service and its dependents."""
        # Cleanup dependents first
        for dependent in self.dependents:
            if dependent.is_healthy():
                await dependent.cleanup()
        
        # Cleanup this service
        await self._cleanup_service()
    
    @abstractmethod
    async def _cleanup_service(self) -> None:
        """Cleanup the service (to be implemented by subclasses)."""
        pass

class BaseEngine(BaseComponent):
    """
    Base engine class for all engines.
    
    Extends BaseComponent with engine-specific functionality.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.config = {}
        self.metrics = {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the engine."""
        self.config.update(config)
        self.metadata.update(config)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        return self.metrics.copy()
    
    def update_metric(self, name: str, value: Any) -> None:
        """Update a metric."""
        self.metrics[name] = value
        self.last_updated = datetime.utcnow()

class BaseManager(BaseComponent):
    """
    Base manager class for all managers.
    
    Extends BaseComponent with manager-specific functionality.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.managed_components = {}
    
    def register_component(self, component: BaseComponent) -> None:
        """Register a component to be managed."""
        self.managed_components[component.name] = component
    
    def unregister_component(self, name: str) -> None:
        """Unregister a component."""
        if name in self.managed_components:
            del self.managed_components[name]
    
    def get_component(self, name: str) -> Optional[BaseComponent]:
        """Get a managed component."""
        return self.managed_components.get(name)
    
    def get_all_components(self) -> Dict[str, BaseComponent]:
        """Get all managed components."""
        return self.managed_components.copy()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup all managed components."""
        for component in self.managed_components.values():
            if component.is_healthy():
                await component.cleanup()

class ComponentRegistry:
    """
    Global component registry.
    
    Manages all system components and their dependencies.
    """
    
    def __init__(self):
        self.components = {}
        self.dependencies = {}
    
    def register(self, component: BaseComponent) -> None:
        """Register a component."""
        self.components[component.name] = component
        logger.info(f"Registered component: {component.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a component."""
        if name in self.components:
            del self.components[name]
            logger.info(f"Unregistered component: {name}")
    
    def get(self, name: str) -> Optional[BaseComponent]:
        """Get a component by name."""
        return self.components.get(name)
    
    def get_all(self) -> Dict[str, BaseComponent]:
        """Get all components."""
        return self.components.copy()
    
    def get_healthy_components(self) -> Dict[str, BaseComponent]:
        """Get all healthy components."""
        return {
            name: component for name, component in self.components.items()
            if component.is_healthy()
        }
    
    async def initialize_all(self) -> None:
        """Initialize all components."""
        for component in self.components.values():
            if not component.is_healthy():
                await component.initialize()
    
    async def cleanup_all(self) -> None:
        """Cleanup all components."""
        for component in self.components.values():
            if component.is_healthy():
                await component.cleanup()

# Global component registry
component_registry = ComponentRegistry()











