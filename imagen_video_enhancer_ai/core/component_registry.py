"""
Component Registry
==================

Registry for managing application components.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from abc import ABC

from .lifecycle import LifecycleManager, LifecycleState
from .dependency_injection import DependencyContainer

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for application components."""
    
    def __init__(self, container: Optional[DependencyContainer] = None):
        """
        Initialize component registry.
        
        Args:
            container: Optional dependency injection container
        """
        self.container = container or DependencyContainer()
        self.components: Dict[str, Any] = {}
        self.lifecycle_managers: Dict[str, LifecycleManager] = {}
    
    def register(
        self,
        name: str,
        component: Any,
        lifecycle: bool = True
    ):
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component instance
            lifecycle: Whether to manage lifecycle
        """
        self.components[name] = component
        
        # Register in DI container
        self.container.register(name, instance=component)
        
        # Setup lifecycle if needed
        if lifecycle:
            if hasattr(component, 'lifecycle'):
                self.lifecycle_managers[name] = component.lifecycle
            else:
                lifecycle_manager = LifecycleManager(name)
                self.lifecycle_managers[name] = lifecycle_manager
        
        logger.info(f"Registered component: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
        """
        if name in self.components:
            return self.components[name]
        
        # Try DI container
        try:
            return self.container.get(name)
        except ValueError:
            raise ValueError(f"Component not found: {name}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all components."""
        return self.components.copy()
    
    async def initialize_all(self):
        """Initialize all components with lifecycle."""
        for name, lifecycle in self.lifecycle_managers.items():
            if lifecycle.get_state() == LifecycleState.UNINITIALIZED:
                await lifecycle.initialize()
    
    async def start_all(self):
        """Start all components with lifecycle."""
        for name, lifecycle in self.lifecycle_managers.items():
            state = lifecycle.get_state()
            if state == LifecycleState.INITIALIZED:
                await lifecycle.start()
            elif state == LifecycleState.UNINITIALIZED:
                await lifecycle.initialize()
                await lifecycle.start()
    
    async def stop_all(self):
        """Stop all components with lifecycle."""
        for name, lifecycle in self.lifecycle_managers.items():
            if lifecycle.get_state() == LifecycleState.RUNNING:
                await lifecycle.stop()
    
    async def shutdown_all(self):
        """Shutdown all components with lifecycle."""
        for name, lifecycle in self.lifecycle_managers.items():
            await lifecycle.shutdown()
    
    def get_container(self) -> DependencyContainer:
        """Get dependency injection container."""
        return self.container




