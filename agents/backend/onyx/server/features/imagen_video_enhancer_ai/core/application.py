"""
Application Core
================

Core application setup and lifecycle management.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .component_registry import ComponentRegistry
from .dependency_injection import DependencyContainer
from .lifecycle import LifecycleManager
from .config_manager import ConfigManager
from .enhancer_agent import EnhancerAgent
from ..config.enhancer_config import EnhancerConfig

logger = logging.getLogger(__name__)


class Application:
    """Main application class with lifecycle management."""
    
    def __init__(
        self,
        config: Optional[EnhancerConfig] = None,
        config_file: Optional[Path] = None
    ):
        """
        Initialize application.
        
        Args:
            config: Optional enhancer config
            config_file: Optional config file path
        """
        self.config = config or EnhancerConfig()
        self.config_file = config_file
        
        # Initialize core components
        self.container = DependencyContainer()
        self.registry = ComponentRegistry(self.container)
        self.lifecycle = LifecycleManager("Application")
        
        # Config manager
        self.config_manager = ConfigManager(config_file)
        
        # Register lifecycle hooks
        self._setup_lifecycle_hooks()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_lifecycle_hooks(self):
        """Setup lifecycle hooks."""
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
    
    def _initialize_components(self):
        """Initialize application components."""
        # Register config manager
        self.registry.register("config_manager", self.config_manager)
        
        # Register enhancer agent
        agent = EnhancerAgent(config=self.config)
        self.registry.register("agent", agent)
        
        logger.info("Application components initialized")
    
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
    
    def get_agent(self) -> EnhancerAgent:
        """Get enhancer agent."""
        return self.get_component("agent")
    
    def get_config_manager(self) -> ConfigManager:
        """Get config manager."""
        return self.get_component("config_manager")
    
    def get_container(self) -> DependencyContainer:
        """Get dependency injection container."""
        return self.container
    
    def get_registry(self) -> ComponentRegistry:
        """Get component registry."""
        return self.registry




