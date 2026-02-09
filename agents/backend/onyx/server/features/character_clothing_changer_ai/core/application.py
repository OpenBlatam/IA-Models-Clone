"""
Application Core
================

Core application setup and lifecycle management.
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Import dependencies if available
try:
    from .component_registry import ComponentRegistry
    from .dependency_injection import DependencyContainer
    from .lifecycle import LifecycleManager
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    # Fallback classes
    class ComponentRegistry:
        def __init__(self, container=None):
            self.components = {}
        def register(self, name, component):
            self.components[name] = component
        def get(self, name):
            return self.components.get(name)
        async def initialize_all(self):
            pass
        async def start_all(self):
            pass
        async def stop_all(self):
            pass
        async def shutdown_all(self):
            pass
    
    class DependencyContainer:
        pass
    
    class LifecycleManager:
        def __init__(self, name):
            self.name = name
            self.hooks = {}
        def register_hook(self, stage, hook, priority=0):
            if stage not in self.hooks:
                self.hooks[stage] = []
            self.hooks[stage].append((priority, hook))
        async def initialize(self):
            pass
        async def start(self):
            pass
        async def stop(self):
            pass
        async def shutdown(self):
            pass


class Application:
    """
    Main application class with lifecycle management.
    
    Features:
    - Component registry
    - Dependency injection
    - Lifecycle management
    - Configuration management
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Path] = None
    ):
        """
        Initialize application.
        
        Args:
            config: Optional application config
            config_file: Optional config file path
        """
        self.config = config or {}
        self.config_file = config_file
        
        # Initialize core components
        self.container = DependencyContainer()
        self.registry = ComponentRegistry(self.container)
        self.lifecycle = LifecycleManager("Application")
        
        # Register lifecycle hooks
        self._setup_lifecycle_hooks()
    
    def _setup_lifecycle_hooks(self):
        """Setup lifecycle hooks."""
        if hasattr(self.lifecycle, 'register_hook'):
            self.lifecycle.register_hook(
                "before_init",
                self._before_init,
                priority=0
            )
            self.lifecycle.register_hook(
                "after_init",
                self._after_init,
                priority=0
            )
            self.lifecycle.register_hook(
                "before_start",
                self._before_start,
                priority=0
            )
            self.lifecycle.register_hook(
                "after_start",
                self._after_start,
                priority=0
            )
            self.lifecycle.register_hook(
                "before_stop",
                self._before_stop,
                priority=0
            )
            self.lifecycle.register_hook(
                "after_stop",
                self._after_stop,
                priority=0
            )
    
    async def _before_init(self):
        """Before initialization hook."""
        logger.info("Application: Before initialization")
    
    async def _after_init(self):
        """After initialization hook."""
        logger.info("Application: After initialization")
    
    async def _before_start(self):
        """Before start hook."""
        logger.info("Application: Before start")
    
    async def _after_start(self):
        """After start hook."""
        logger.info("Application: After start")
    
    async def _before_stop(self):
        """Before stop hook."""
        logger.info("Application: Before stop")
    
    async def _after_stop(self):
        """After stop hook."""
        logger.info("Application: After stop")
    
    def register_component(self, name: str, component: Any):
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.registry.register(name, component)
        logger.info(f"Registered component: {name}")
    
    async def initialize(self):
        """Initialize application."""
        await self.lifecycle.initialize()
        await self.registry.initialize_all()
    
    async def start(self):
        """Start application."""
        await self.lifecycle.start()
        await self.registry.start_all()
    
    async def stop(self):
        """Stop application."""
        await self.registry.stop_all()
        await self.lifecycle.stop()
    
    async def shutdown(self):
        """Shutdown application."""
        await self.registry.shutdown_all()
        await self.lifecycle.shutdown()
    
    def get_component(self, name: str) -> Any:
        """
        Get component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
        """
        return self.registry.get(name)
    
    def get_container(self) -> DependencyContainer:
        """Get dependency injection container."""
        return self.container
    
    def get_registry(self) -> ComponentRegistry:
        """Get component registry."""
        return self.registry

