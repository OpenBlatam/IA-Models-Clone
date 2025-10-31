from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Type, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from ..core.exceptions import PluginError, DependencyError, ValidationError
from ..core.types import PluginInfo
from .base import BasePlugin
from .loader import PluginLoader
from .validator import PluginValidator, ValidationLevel
from typing import Any, List, Dict, Optional
"""
Plugin Registry - Production-Ready Plugin Management System

This module provides a comprehensive plugin registry with:
- Plugin lifecycle management
- Dependency resolution
- Event handling
- Configuration management
- Performance monitoring
"""



logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin states in the registry."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RegistryEntry:
    """Entry in the plugin registry."""
    plugin: BasePlugin
    info: PluginInfo
    state: PluginState
    config: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    load_time: float = 0.0
    init_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


class PluginRegistry:
    """
    Production-ready plugin registry with comprehensive management capabilities.
    
    Features:
    - Plugin lifecycle management
    - Dependency resolution
    - Event handling
    - Configuration management
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(
        self,
        auto_discover: bool = True,
        auto_load: bool = False,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ):
        
    """__init__ function."""
self.auto_discover = auto_discover
        self.auto_load = auto_load
        self.validation_level = validation_level
        
        # Core components
        self.loader = PluginLoader()
        self.validator = PluginValidator(validation_level)
        
        # Registry state
        self.plugins: Dict[str, RegistryEntry] = {}
        self.discovered_plugins: Dict[str, PluginInfo] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_plugins': 0,
            'loaded_plugins': 0,
            'initialized_plugins': 0,
            'running_plugins': 0,
            'failed_plugins': 0,
            'total_load_time': 0.0,
            'total_init_time': 0.0
        }
        
        logger.info(f"PluginRegistry initialized (auto_discover={auto_discover}, auto_load={auto_load})")
    
    async def discover_plugins(self, plugin_dirs: Optional[List[str]] = None) -> List[PluginInfo]:
        """
        Discover all available plugins.
        
        Args:
            plugin_dirs: Directories to search for plugins
            
        Returns:
            List of discovered plugin information
        """
        logger.info("🔍 Starting plugin discovery...")
        
        try:
            discovered = await self.loader.discover_plugins(plugin_dirs)
            
            # Store discovered plugins
            for plugin_info in discovered:
                self.discovered_plugins[plugin_info.name] = plugin_info
            
            # Validate discovered plugins
            for plugin_info in discovered:
                validation_report = self.validator.validate_plugin_info(plugin_info)
                if validation_report.overall_status.value == "failed":
                    logger.warning(f"⚠️ Plugin '{plugin_info.name}' failed validation")
            
            logger.info(f"✅ Discovered {len(discovered)} plugins")
            
            # Trigger discovery event
            await self._trigger_event("plugins_discovered", discovered)
            
            return discovered
            
        except Exception as e:
            logger.error(f"❌ Plugin discovery failed: {e}")
            raise PluginError(f"Plugin discovery failed: {e}")
    
    async def load_plugin(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]] = None,
        force_reload: bool = False
    ) -> BasePlugin:
        """
        Load a plugin into the registry.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Plugin configuration
            force_reload: Whether to force reload if already loaded
            
        Returns:
            Loaded plugin instance
        """
        if plugin_name in self.plugins and not force_reload:
            logger.info(f"Plugin '{plugin_name}' already loaded")
            return self.plugins[plugin_name].plugin
        
        logger.info(f"📦 Loading plugin: {plugin_name}")
        
        try:
            # Load the plugin
            load_result = await self.loader.load_plugin(plugin_name, config)
            
            if not load_result.success:
                raise PluginError(f"Failed to load plugin '{plugin_name}': {load_result.error}")
            
            plugin = load_result.plugin
            
            # Validate the plugin
            validation_report = self.validator.validate_plugin_detailed(plugin)
            if validation_report.overall_status.value == "failed":
                raise ValidationError(f"Plugin '{plugin_name}' failed validation")
            
            # Get plugin info
            plugin_info = plugin.get_metadata()
            
            # Create registry entry
            entry = RegistryEntry(
                plugin=plugin,
                info=plugin_info,
                state=PluginState.LOADED,
                config=config or {},
                load_time=load_result.load_time
            )
            
            # Store in registry
            self.plugins[plugin_name] = entry
            
            # Update statistics
            self.stats['loaded_plugins'] += 1
            self.stats['total_load_time'] += load_result.load_time
            
            logger.info(f"✅ Plugin '{plugin_name}' loaded successfully")
            
            # Trigger load event
            await self._trigger_event("plugin_loaded", plugin_name, plugin)
            
            return plugin
            
        except Exception as e:
            logger.error(f"❌ Failed to load plugin '{plugin_name}': {e}")
            self.stats['failed_plugins'] += 1
            raise
    
    async def initialize_plugin(self, plugin_name: str) -> BasePlugin:
        """
        Initialize a loaded plugin.
        
        Args:
            plugin_name: Name of the plugin to initialize
            
        Returns:
            Initialized plugin instance
        """
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin '{plugin_name}' not loaded")
        
        entry = self.plugins[plugin_name]
        
        if entry.state == PluginState.INITIALIZED:
            logger.info(f"Plugin '{plugin_name}' already initialized")
            return entry.plugin
        
        logger.info(f"🚀 Initializing plugin: {plugin_name}")
        
        try:
            start_time = time.time()
            
            # Initialize the plugin
            await entry.plugin.initialize()
            
            init_time = time.time() - start_time
            
            # Update entry
            entry.state = PluginState.INITIALIZED
            entry.init_time = init_time
            
            # Update statistics
            self.stats['initialized_plugins'] += 1
            self.stats['total_init_time'] += init_time
            
            logger.info(f"✅ Plugin '{plugin_name}' initialized successfully")
            
            # Trigger initialization event
            await self._trigger_event("plugin_initialized", plugin_name, entry.plugin)
            
            return entry.plugin
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize plugin '{plugin_name}': {e}")
            entry.state = PluginState.ERROR
            entry.last_error = str(e)
            entry.error_count += 1
            raise
    
    async def start_plugin(self, plugin_name: str) -> BasePlugin:
        """
        Start a plugin (if it has a start method).
        
        Args:
            plugin_name: Name of the plugin to start
            
        Returns:
            Started plugin instance
        """
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin '{plugin_name}' not loaded")
        
        entry = self.plugins[plugin_name]
        
        if entry.state == PluginState.RUNNING:
            logger.info(f"Plugin '{plugin_name}' already running")
            return entry.plugin
        
        if entry.state != PluginState.INITIALIZED:
            await self.initialize_plugin(plugin_name)
        
        logger.info(f"▶️ Starting plugin: {plugin_name}")
        
        try:
            # Start the plugin if it has a start method
            if hasattr(entry.plugin, 'start') and callable(entry.plugin.start):
                await entry.plugin.start()
            
            entry.state = PluginState.RUNNING
            
            # Update statistics
            self.stats['running_plugins'] += 1
            
            logger.info(f"✅ Plugin '{plugin_name}' started successfully")
            
            # Trigger start event
            await self._trigger_event("plugin_started", plugin_name, entry.plugin)
            
            return entry.plugin
            
        except Exception as e:
            logger.error(f"❌ Failed to start plugin '{plugin_name}': {e}")
            entry.state = PluginState.ERROR
            entry.last_error = str(e)
            entry.error_count += 1
            raise
    
    async def stop_plugin(self, plugin_name: str) -> BasePlugin:
        """
        Stop a plugin (if it has a stop method).
        
        Args:
            plugin_name: Name of the plugin to stop
            
        Returns:
            Stopped plugin instance
        """
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin '{plugin_name}' not loaded")
        
        entry = self.plugins[plugin_name]
        
        if entry.state == PluginState.STOPPED:
            logger.info(f"Plugin '{plugin_name}' already stopped")
            return entry.plugin
        
        logger.info(f"⏹️ Stopping plugin: {plugin_name}")
        
        try:
            # Stop the plugin if it has a stop method
            if hasattr(entry.plugin, 'stop') and callable(entry.plugin.stop):
                await entry.plugin.stop()
            
            entry.state = PluginState.STOPPED
            
            # Update statistics
            if entry.state == PluginState.RUNNING:
                self.stats['running_plugins'] -= 1
            
            logger.info(f"✅ Plugin '{plugin_name}' stopped successfully")
            
            # Trigger stop event
            await self._trigger_event("plugin_stopped", plugin_name, entry.plugin)
            
            return entry.plugin
            
        except Exception as e:
            logger.error(f"❌ Failed to stop plugin '{plugin_name}': {e}")
            entry.last_error = str(e)
            entry.error_count += 1
            raise
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin from the registry.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin was unloaded successfully
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' not in registry")
            return False
        
        entry = self.plugins[plugin_name]
        
        logger.info(f"🗑️ Unloading plugin: {plugin_name}")
        
        try:
            # Stop the plugin first
            if entry.state == PluginState.RUNNING:
                await self.stop_plugin(plugin_name)
            
            # Cleanup the plugin
            if hasattr(entry.plugin, 'cleanup') and callable(entry.plugin.cleanup):
                await entry.plugin.cleanup()
            
            # Remove from registry
            del self.plugins[plugin_name]
            
            # Update statistics
            self.stats['loaded_plugins'] -= 1
            if entry.state == PluginState.INITIALIZED:
                self.stats['initialized_plugins'] -= 1
            if entry.state == PluginState.RUNNING:
                self.stats['running_plugins'] -= 1
            
            logger.info(f"✅ Plugin '{plugin_name}' unloaded successfully")
            
            # Trigger unload event
            await self._trigger_event("plugin_unloaded", plugin_name)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload plugin '{plugin_name}': {e}")
            return False
    
    async def load_all_plugins(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
        """
        Load all discovered plugins.
        
        Args:
            configs: Configuration for each plugin
            
        Returns:
            List of successfully loaded plugin names
        """
        if not self.discovered_plugins:
            await self.discover_plugins()
        
        configs = configs or {}
        loaded_plugins = []
        
        logger.info(f"📦 Loading {len(self.discovered_plugins)} discovered plugins...")
        
        for plugin_name, plugin_info in self.discovered_plugins.items():
            try:
                config = configs.get(plugin_name, {})
                await self.load_plugin(plugin_name, config)
                loaded_plugins.append(plugin_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to load plugin '{plugin_name}': {e}")
        
        logger.info(f"✅ Successfully loaded {len(loaded_plugins)} plugins")
        return loaded_plugins
    
    async def initialize_all_plugins(self) -> List[str]:
        """
        Initialize all loaded plugins.
        
        Returns:
            List of successfully initialized plugin names
        """
        initialized_plugins = []
        
        logger.info(f"🚀 Initializing {len(self.plugins)} loaded plugins...")
        
        for plugin_name in list(self.plugins.keys()):
            try:
                await self.initialize_plugin(plugin_name)
                initialized_plugins.append(plugin_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize plugin '{plugin_name}': {e}")
        
        logger.info(f"✅ Successfully initialized {len(initialized_plugins)} plugins")
        return initialized_plugins
    
    async def start_all_plugins(self) -> List[str]:
        """
        Start all initialized plugins.
        
        Returns:
            List of successfully started plugin names
        """
        started_plugins = []
        
        logger.info(f"▶️ Starting {len(self.plugins)} plugins...")
        
        for plugin_name in list(self.plugins.keys()):
            try:
                await self.start_plugin(plugin_name)
                started_plugins.append(plugin_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to start plugin '{plugin_name}': {e}")
        
        logger.info(f"✅ Successfully started {len(started_plugins)} plugins")
        return started_plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        entry = self.plugins.get(plugin_name)
        return entry.plugin if entry else None
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information by name."""
        entry = self.plugins.get(plugin_name)
        return entry.info if entry else None
    
    def get_plugin_state(self, plugin_name: str) -> Optional[PluginState]:
        """Get plugin state by name."""
        entry = self.plugins.get(plugin_name)
        return entry.state if entry else None
    
    def list_plugins(self, state: Optional[PluginState] = None) -> List[str]:
        """List plugin names, optionally filtered by state."""
        if state is None:
            return list(self.plugins.keys())
        
        return [
            name for name, entry in self.plugins.items()
            if entry.state == state
        ]
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        stats = self.stats.copy()
        
        # Add current state counts
        state_counts = defaultdict(int)
        for entry in self.plugins.values():
            state_counts[entry.state.value] += 1
        
        stats['state_counts'] = dict(state_counts)
        stats['total_plugins'] = len(self.plugins)
        
        return stats
    
    def add_event_handler(self, event: str, handler: Callable):
        """Add an event handler."""
        self.event_handlers[event].append(handler)
        logger.debug(f"Added event handler for '{event}'")
    
    def remove_event_handler(self, event: str, handler: Callable):
        """Remove an event handler."""
        if event in self.event_handlers:
            self.event_handlers[event] = [
                h for h in self.event_handlers[event] if h != handler
            ]
            logger.debug(f"Removed event handler for '{event}'")
    
    async def _trigger_event(self, event: str, *args, **kwargs):
        """Trigger an event with all registered handlers."""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*args, **kwargs)
                    else:
                        handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Event handler error for '{event}': {e}")
    
    async def shutdown(self) -> Any:
        """Shutdown the registry and all plugins."""
        logger.info("🔄 Shutting down plugin registry...")
        
        # Stop all running plugins
        running_plugins = self.list_plugins(PluginState.RUNNING)
        for plugin_name in running_plugins:
            try:
                await self.stop_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Error stopping plugin '{plugin_name}': {e}")
        
        # Unload all plugins
        loaded_plugins = list(self.plugins.keys())
        for plugin_name in loaded_plugins:
            try:
                await self.unload_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Error unloading plugin '{plugin_name}': {e}")
        
        logger.info("✅ Plugin registry shutdown complete") 