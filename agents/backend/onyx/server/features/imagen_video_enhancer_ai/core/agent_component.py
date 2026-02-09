"""
Agent Component
================

Base component system for agent architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentConfig:
    """Component configuration."""
    name: str
    enabled: bool = True
    auto_start: bool = True
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Component health status."""
    status: ComponentStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


class AgentComponent(ABC):
    """Base component for agent architecture."""
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize component.
        
        Args:
            config: Component configuration
        """
        self.config = config
        self.status = ComponentStatus.UNINITIALIZED
        self.health: Optional[ComponentHealth] = None
        self.stats = {
            "initializations": 0,
            "starts": 0,
            "stops": 0,
            "errors": 0
        }
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """Component-specific initialization."""
        pass
    
    @abstractmethod
    async def _do_start(self) -> None:
        """Component-specific start logic."""
        pass
    
    @abstractmethod
    async def _do_stop(self) -> None:
        """Component-specific stop logic."""
        pass
    
    async def initialize(self) -> None:
        """Initialize component."""
        async with self._lock:
            if self.status != ComponentStatus.UNINITIALIZED:
                return
            
            try:
                self.status = ComponentStatus.INITIALIZING
                await self._do_initialize()
                self.status = ComponentStatus.READY
                self.stats["initializations"] += 1
                logger.info(f"Component {self.config.name} initialized")
            except Exception as e:
                self.status = ComponentStatus.ERROR
                self.stats["errors"] += 1
                logger.error(f"Error initializing component {self.config.name}: {e}")
                raise
    
    async def start(self) -> None:
        """Start component."""
        async with self._lock:
            if self.status == ComponentStatus.UNINITIALIZED:
                await self.initialize()
            
            if self.status != ComponentStatus.READY:
                raise RuntimeError(f"Component {self.config.name} not ready (status: {self.status.value})")
            
            try:
                self.status = ComponentStatus.RUNNING
                await self._do_start()
                self.stats["starts"] += 1
                logger.info(f"Component {self.config.name} started")
            except Exception as e:
                self.status = ComponentStatus.ERROR
                self.stats["errors"] += 1
                logger.error(f"Error starting component {self.config.name}: {e}")
                raise
    
    async def stop(self) -> None:
        """Stop component."""
        async with self._lock:
            if self.status == ComponentStatus.STOPPED:
                return
            
            try:
                await self._do_stop()
                self.status = ComponentStatus.STOPPED
                self.stats["stops"] += 1
                logger.info(f"Component {self.config.name} stopped")
            except Exception as e:
                self.status = ComponentStatus.ERROR
                self.stats["errors"] += 1
                logger.error(f"Error stopping component {self.config.name}: {e}")
    
    async def pause(self) -> None:
        """Pause component."""
        async with self._lock:
            if self.status == ComponentStatus.RUNNING:
                self.status = ComponentStatus.PAUSED
                logger.info(f"Component {self.config.name} paused")
    
    async def resume(self) -> None:
        """Resume component."""
        async with self._lock:
            if self.status == ComponentStatus.PAUSED:
                self.status = ComponentStatus.RUNNING
                logger.info(f"Component {self.config.name} resumed")
    
    async def health_check(self) -> ComponentHealth:
        """
        Perform health check.
        
        Returns:
            Component health status
        """
        health = ComponentHealth(
            status=self.status,
            message=f"Component {self.config.name} is {self.status.value}",
            metrics=self.get_stats()
        )
        self.health = health
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            **self.stats
        }
    
    def is_ready(self) -> bool:
        """Check if component is ready."""
        return self.status == ComponentStatus.READY
    
    def is_running(self) -> bool:
        """Check if component is running."""
        return self.status == ComponentStatus.RUNNING


class ComponentManager:
    """Manager for agent components."""
    
    def __init__(self):
        """Initialize component manager."""
        self.components: Dict[str, AgentComponent] = {}
        self.initialization_order: List[str] = []
    
    def register(self, component: AgentComponent):
        """
        Register a component.
        
        Args:
            component: Component instance
        """
        self.components[component.config.name] = component
        logger.debug(f"Registered component: {component.config.name}")
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate initialization order based on dependencies."""
        # Topological sort
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self.components:
                component = self.components[name]
                for dep in component.config.dependencies:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        for name in self.components.keys():
            if name not in visited:
                visit(name)
        
        return order
    
    async def initialize_all(self):
        """Initialize all components in dependency order."""
        self.initialization_order = self._calculate_initialization_order()
        
        for name in self.initialization_order:
            component = self.components[name]
            if component.config.enabled:
                await component.initialize()
                if component.config.auto_start:
                    await component.start()
    
    async def start_all(self):
        """Start all components."""
        for name in self.initialization_order:
            component = self.components[name]
            if component.config.enabled and component.is_ready():
                await component.start()
    
    async def stop_all(self):
        """Stop all components in reverse order."""
        for name in reversed(self.initialization_order):
            component = self.components[name]
            if component.is_running():
                await component.stop()
    
    async def health_check_all(self) -> Dict[str, ComponentHealth]:
        """Perform health check on all components."""
        health_status = {}
        for name, component in self.components.items():
            health_status[name] = await component.health_check()
        return health_status
    
    def get_component(self, name: str) -> Optional[AgentComponent]:
        """Get component by name."""
        return self.components.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all components."""
        return {
            name: component.get_stats()
            for name, component in self.components.items()
        }




