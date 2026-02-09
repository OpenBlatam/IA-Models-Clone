from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from ..core.exceptions import PluginError, ValidationError
from ..core.types import PluginInfo
from .base import BasePlugin
from .registry import PluginRegistry, PluginState
from .validator import ValidationLevel
from typing import Any, List, Dict, Optional
"""
Plugin Manager - High-Level Plugin Management System

This module provides a user-friendly interface for plugin management with:
- Simple API for common operations
- Auto-discovery and loading
- Dependency resolution
- Configuration management
- Event handling
- Performance monitoring
"""



logger = logging.getLogger(__name__)


@dataclass
class ManagerConfig:
    """Configuration for the plugin manager."""
    auto_discover: bool = True
    auto_load: bool = False
    auto_initialize: bool = False
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    plugin_dirs: List[str] = None
    default_config: Dict[str, Any] = None
    enable_events: bool = True
    enable_metrics: bool = True
    
    def __post_init__(self) -> Any:
        if self.plugin_dirs is None:
            self.plugin_dirs = [
                "./plugins",
                "./ai_video/plugins",
                "./extensions",
                "~/.ai_video/plugins"
            ]
        if self.default_config is None:
            self.default_config = {}


class PluginManager:
    """
    High-level plugin manager with user-friendly API.
    
    Features:
    - Simple, intuitive API
    - Auto-discovery and loading
    - Dependency resolution
    - Configuration management
    - Event handling
    - Performance monitoring
    - Error recovery
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        
    """__init__ function."""
self.config = config or ManagerConfig()
        self.registry = PluginRegistry(
            auto_discover=self.config.auto_discover,
            auto_load=self.config.auto_load,
            validation_level=self.config.validation_level
        )
        
        # Setup event handlers if enabled
        if self.config.enable_events:
            self._setup_event_handlers()
        
        logger.info("🎯 PluginManager initialized")
    
    async def start(self) -> bool:
        """
        Start the plugin manager and discover/load plugins.
        
        Returns:
            True if startup was successful
        """
        logger.info("🚀 Starting PluginManager...")
        
        try:
            # Discover plugins
            if self.config.auto_discover:
                await self.discover_plugins()
            
            # Load plugins if auto-load is enabled
            if self.config.auto_load:
                await self.load_all_plugins()
            
            # Initialize plugins if auto-initialize is enabled
            if self.config.auto_initialize:
                await self.initialize_all_plugins()
            
            logger.info("✅ PluginManager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start PluginManager: {e}")
            return False
    
    async def discover_plugins(self, plugin_dirs: Optional[List[str]] = None) -> List[PluginInfo]:
        """
        Discover available plugins.
        
        Args:
            plugin_dirs: Directories to search (uses config if None)
            
        Returns:
            List of discovered plugin information
        """
        dirs = plugin_dirs or self.config.plugin_dirs
        return await self.registry.discover_plugins(dirs)
    
    async def load_plugin(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = None
    ) -> BasePlugin:
        """
        Load a plugin with simple configuration.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Plugin configuration (merged with default config)
            auto_initialize: Whether to auto-initialize (uses config if None)
            
        Returns:
            Loaded plugin instance
        """
        # Merge with default config
        final_config = self.config.default_config.copy()
        if config:
            final_config.update(config)
        
        # Load the plugin
        plugin = await self.registry.load_plugin(plugin_name, final_config)
        
        # Auto-initialize if requested
        if auto_initialize is None:
            auto_initialize = self.config.auto_initialize
        
        if auto_initialize:
            await self.initialize_plugin(plugin_name)
        
        return plugin
    
    async def load_all_plugins(
        self, 
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_initialize: bool = None
    ) -> List[str]:
        """
        Load all discovered plugins.
        
        Args:
            configs: Configuration for each plugin
            auto_initialize: Whether to auto-initialize (uses config if None)
            
        Returns:
            List of successfully loaded plugin names
        """
        # Merge with default configs
        final_configs = {}
        if configs:
            for name, config in configs.items():
                final_configs[name] = self.config.default_config.copy()
                final_configs[name].update(config)
        
        # Load plugins
        loaded_plugins = await self.registry.load_all_plugins(final_configs)
        
        # Auto-initialize if requested
        if auto_initialize is None:
            auto_initialize = self.config.auto_initialize
        
        if auto_initialize:
            await self.initialize_all_plugins()
        
        return loaded_plugins
    
    async def initialize_plugin(self, plugin_name: str) -> BasePlugin:
        """Initialize a plugin."""
        return await self.registry.initialize_plugin(plugin_name)
    
    async def initialize_all_plugins(self) -> List[str]:
        """Initialize all loaded plugins."""
        return await self.registry.initialize_all_plugins()
    
    async def start_plugin(self, plugin_name: str) -> BasePlugin:
        """Start a plugin."""
        return await self.registry.start_plugin(plugin_name)
    
    async def start_all_plugins(self) -> List[str]:
        """Start all initialized plugins."""
        return await self.registry.start_all_plugins()
    
    async def stop_plugin(self, plugin_name: str) -> BasePlugin:
        """Stop a plugin."""
        return await self.registry.stop_plugin(plugin_name)
    
    async def stop_all_plugins(self) -> List[str]:
        """Stop all running plugins."""
        stopped_plugins = []
        
        running_plugins = self.registry.list_plugins(PluginState.RUNNING)
        for plugin_name in running_plugins:
            try:
                await self.registry.stop_plugin(plugin_name)
                stopped_plugins.append(plugin_name)
            except Exception as e:
                logger.error(f"Failed to stop plugin '{plugin_name}': {e}")
        
        return stopped_plugins
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        return await self.registry.unload_plugin(plugin_name)
    
    async def unload_all_plugins(self) -> List[str]:
        """Unload all plugins."""
        unloaded_plugins = []
        
        loaded_plugins = list(self.registry.plugins.keys())
        for plugin_name in loaded_plugins:
            try:
                if await self.registry.unload_plugin(plugin_name):
                    unloaded_plugins.append(plugin_name)
            except Exception as e:
                logger.error(f"Failed to unload plugin '{plugin_name}': {e}")
        
        return unloaded_plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.registry.get_plugin(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information by name."""
        return self.registry.get_plugin_info(plugin_name)
    
    def get_plugin_state(self, plugin_name: str) -> Optional[PluginState]:
        """Get plugin state by name."""
        return self.registry.get_plugin_state(plugin_name)
    
    def list_plugins(self, state: Optional[PluginState] = None) -> List[str]:
        """List plugin names, optionally filtered by state."""
        return self.registry.list_plugins(state)
    
    def list_plugins_by_category(self, category: str) -> List[str]:
        """List plugins by category."""
        plugins = []
        for name, entry in self.registry.plugins.items():
            if entry.info.category == category:
                plugins.append(name)
        return plugins
    
    def get_plugin_config(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin configuration."""
        entry = self.registry.plugins.get(plugin_name)
        return entry.config if entry else None
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        entry = self.registry.plugins.get(plugin_name)
        if entry:
            entry.config.update(config)
            logger.info(f"Updated configuration for plugin '{plugin_name}'")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.registry.get_plugin_stats()
        
        # Add manager-specific stats
        stats.update({
            'manager_config': {
                'auto_discover': self.config.auto_discover,
                'auto_load': self.config.auto_load,
                'auto_initialize': self.config.auto_initialize,
                'validation_level': self.config.validation_level.value,
                'plugin_dirs': self.config.plugin_dirs,
                'enable_events': self.config.enable_events,
                'enable_metrics': self.config.enable_metrics
            },
            'discovered_plugins': len(self.registry.discovered_plugins),
            'total_plugins': len(self.registry.plugins)
        })
        
        return stats
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get a health report for all plugins."""
        health_report = {
            'overall_status': 'healthy',
            'total_plugins': len(self.registry.plugins),
            'healthy_plugins': 0,
            'unhealthy_plugins': 0,
            'plugin_details': {}
        }
        
        for name, entry in self.registry.plugins.items():
            plugin_health = {
                'state': entry.state.value,
                'error_count': entry.error_count,
                'last_error': entry.last_error,
                'load_time': entry.load_time,
                'init_time': entry.init_time
            }
            
            # Determine health status
            if entry.state == PluginState.ERROR or entry.error_count > 0:
                plugin_health['status'] = 'unhealthy'
                health_report['unhealthy_plugins'] += 1
                health_report['overall_status'] = 'unhealthy'
            else:
                plugin_health['status'] = 'healthy'
                health_report['healthy_plugins'] += 1
            
            health_report['plugin_details'][name] = plugin_health
        
        return health_report
    
    def add_event_handler(self, event: str, handler: Callable):
        """Add an event handler."""
        if self.config.enable_events:
            self.registry.add_event_handler(event, handler)
    
    def remove_event_handler(self, event: str, handler: Callable):
        """Remove an event handler."""
        if self.config.enable_events:
            self.registry.remove_event_handler(event, handler)
    
    def _setup_event_handlers(self) -> Any:
        """Setup default event handlers."""
        self.registry.add_event_handler("plugin_loaded", self._on_plugin_loaded)
        self.registry.add_event_handler("plugin_initialized", self._on_plugin_initialized)
        self.registry.add_event_handler("plugin_started", self._on_plugin_started)
        self.registry.add_event_handler("plugin_stopped", self._on_plugin_stopped)
        self.registry.add_event_handler("plugin_unloaded", self._on_plugin_unloaded)
    
    async def _on_plugin_loaded(self, plugin_name: str, plugin: BasePlugin):
        """Handle plugin loaded event."""
        logger.info(f"📦 Plugin '{plugin_name}' loaded successfully")
    
    async def _on_plugin_initialized(self, plugin_name: str, plugin: BasePlugin):
        """Handle plugin initialized event."""
        logger.info(f"🚀 Plugin '{plugin_name}' initialized successfully")
    
    async def _on_plugin_started(self, plugin_name: str, plugin: BasePlugin):
        """Handle plugin started event."""
        logger.info(f"▶️ Plugin '{plugin_name}' started successfully")
    
    async def _on_plugin_stopped(self, plugin_name: str, plugin: BasePlugin):
        """Handle plugin stopped event."""
        logger.info(f"⏹️ Plugin '{plugin_name}' stopped successfully")
    
    async def _on_plugin_unloaded(self, plugin_name: str):
        """Handle plugin unloaded event."""
        logger.info(f"🗑️ Plugin '{plugin_name}' unloaded successfully")
    
    async def shutdown(self) -> Any:
        """Shutdown the plugin manager."""
        logger.info("🔄 Shutting down PluginManager...")
        
        # Stop all plugins
        await self.stop_all_plugins()
        
        # Unload all plugins
        await self.unload_all_plugins()
        
        # Shutdown registry
        await self.registry.shutdown()
        
        logger.info("✅ PluginManager shutdown complete")
    
    # Convenience methods for common operations
    
    async def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        """Reload a plugin with new configuration."""
        logger.info(f"🔄 Reloading plugin: {plugin_name}")
        
        # Unload if currently loaded
        if plugin_name in self.registry.plugins:
            await self.unload_plugin(plugin_name)
        
        # Load with new config
        return await self.load_plugin(plugin_name, config)
    
    async def restart_plugin(self, plugin_name: str) -> BasePlugin:
        """Restart a plugin."""
        logger.info(f"🔄 Restarting plugin: {plugin_name}")
        
        # Stop if running
        if self.get_plugin_state(plugin_name) == PluginState.RUNNING:
            await self.stop_plugin(plugin_name)
        
        # Start again
        return await self.start_plugin(plugin_name)
    
    def get_plugin_summary(self) -> Dict[str, Any]:
        """Get a summary of all plugins."""
        summary = {
            'total_plugins': len(self.registry.plugins),
            'by_state': {},
            'by_category': {},
            'plugins': {}
        }
        
        # Count by state
        for entry in self.registry.plugins.values():
            state = entry.state.value
            summary['by_state'][state] = summary['by_state'].get(state, 0) + 1
            
            # Count by category
            category = entry.info.category
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Plugin details
            summary['plugins'][entry.info.name] = {
                'version': entry.info.version,
                'description': entry.info.description,
                'category': entry.info.category,
                'state': entry.state.value,
                'author': entry.info.author
            } 