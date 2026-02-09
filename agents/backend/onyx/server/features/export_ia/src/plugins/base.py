"""
Base plugin system for Export IA.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib
import inspect

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin status enumeration."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    LOADING = "loading"
    ERROR = "error"


class PluginType(Enum):
    """Plugin type enumeration."""
    EXPORT = "export"
    QUALITY = "quality"
    AI = "ai"
    WORKFLOW = "workflow"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    SECURITY = "security"


@dataclass
class PluginInfo:
    """Plugin information."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    status: PluginStatus = PluginStatus.DISABLED
    loaded_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._initialized = False
        self._enabled = False
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin information."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    async def enable(self) -> None:
        """Enable the plugin."""
        if not self._initialized:
            await self.initialize()
        
        self._enabled = True
        self.logger.info(f"Plugin {self.info.name} enabled")
    
    async def disable(self) -> None:
        """Disable the plugin."""
        self._enabled = False
        await self.cleanup()
        self.logger.info(f"Plugin {self.info.name} disabled")
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin."""
        self.config.update(config)
        self.logger.info(f"Plugin {self.info.name} configured")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
    
    async def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin."""
        async with self._lock:
            plugin_name = plugin.info.name
            
            if plugin_name in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} already registered")
                return False
            
            # Check dependencies
            if not await self._check_dependencies(plugin):
                return False
            
            # Register plugin
            self.plugins[plugin_name] = plugin
            
            # Add to type registry
            plugin_type = plugin.info.plugin_type
            if plugin_type not in self.plugin_types:
                self.plugin_types[plugin_type] = []
            self.plugin_types[plugin_type].append(plugin_name)
            
            # Register dependencies
            self.dependencies[plugin_name] = plugin.info.dependencies.copy()
            
            self.logger.info(f"Plugin registered: {plugin_name}")
            return True
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        async with self._lock:
            if plugin_name not in self.plugins:
                return False
            
            plugin = self.plugins[plugin_name]
            
            # Disable plugin if enabled
            if plugin.is_enabled():
                await plugin.disable()
            
            # Remove from type registry
            plugin_type = plugin.info.plugin_type
            if plugin_type in self.plugin_types:
                if plugin_name in self.plugin_types[plugin_type]:
                    self.plugin_types[plugin_type].remove(plugin_name)
            
            # Remove dependencies
            if plugin_name in self.dependencies:
                del self.dependencies[plugin_name]
            
            # Remove plugin
            del self.plugins[plugin_name]
            
            self.logger.info(f"Plugin unregistered: {plugin_name}")
            return True
    
    async def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        async with self._lock:
            return self.plugins.get(plugin_name)
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get plugins by type."""
        async with self._lock:
            plugin_names = self.plugin_types.get(plugin_type, [])
            return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    async def get_enabled_plugins(self, plugin_type: Optional[PluginType] = None) -> List[BasePlugin]:
        """Get enabled plugins, optionally filtered by type."""
        async with self._lock:
            plugins = []
            
            if plugin_type:
                plugin_names = self.plugin_types.get(plugin_type, [])
                plugins = [self.plugins[name] for name in plugin_names if name in self.plugins]
            else:
                plugins = list(self.plugins.values())
            
            return [plugin for plugin in plugins if plugin.is_enabled()]
    
    async def _check_dependencies(self, plugin: BasePlugin) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dependency in plugin.info.dependencies:
            if dependency not in self.plugins:
                self.logger.error(f"Plugin {plugin.info.name} missing dependency: {dependency}")
                return False
            
            dep_plugin = self.plugins[dependency]
            if not dep_plugin.is_enabled():
                self.logger.error(f"Plugin {plugin.info.name} dependency not enabled: {dependency}")
                return False
        
        return True
    
    async def list_plugins(self) -> Dict[str, PluginInfo]:
        """List all registered plugins."""
        async with self._lock:
            return {
                name: plugin.info
                for name, plugin in self.plugins.items()
            }


class PluginManager:
    """Manager for plugin lifecycle and operations."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.plugin_directories: List[str] = []
        self.auto_discovery = True
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        if self.auto_discovery:
            await self.discover_plugins()
        
        self.logger.info("Plugin manager initialized")
    
    async def discover_plugins(self) -> None:
        """Discover plugins in configured directories."""
        for directory in self.plugin_directories:
            await self._discover_plugins_in_directory(directory)
    
    async def _discover_plugins_in_directory(self, directory: str) -> None:
        """Discover plugins in a specific directory."""
        try:
            # This would scan the directory for plugin files
            # and dynamically load them
            pass
        except Exception as e:
            self.logger.error(f"Error discovering plugins in {directory}: {e}")
    
    async def load_plugin(self, plugin_class: Type[BasePlugin], config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin class."""
        try:
            plugin = plugin_class(config)
            return await self.registry.register_plugin(plugin)
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_class.__name__}: {e}")
            return False
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        plugin = await self.registry.get_plugin(plugin_name)
        if not plugin:
            return False
        
        try:
            await plugin.enable()
            return True
        except Exception as e:
            self.logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        plugin = await self.registry.get_plugin(plugin_name)
        if not plugin:
            return False
        
        try:
            await plugin.disable()
            return True
        except Exception as e:
            self.logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False
    
    async def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin."""
        plugin = await self.registry.get_plugin(plugin_name)
        if not plugin:
            return False
        
        try:
            plugin.configure(config)
            return True
        except Exception as e:
            self.logger.error(f"Error configuring plugin {plugin_name}: {e}")
            return False
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        plugin = await self.registry.get_plugin(plugin_name)
        return plugin.info if plugin else None
    
    async def list_plugins(self) -> Dict[str, PluginInfo]:
        """List all plugins."""
        return await self.registry.list_plugins()
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get plugins by type."""
        return await self.registry.get_plugins_by_type(plugin_type)
    
    async def cleanup(self) -> None:
        """Cleanup all plugins."""
        for plugin in self.registry.plugins.values():
            try:
                await plugin.disable()
            except Exception as e:
                self.logger.error(f"Error cleaning up plugin {plugin.info.name}: {e}")
        
        self.logger.info("Plugin manager cleanup completed")


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager