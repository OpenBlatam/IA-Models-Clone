"""
Plugin Manager
==============

Advanced plugin management system.
"""

import logging
import importlib
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    enabled: bool = True


class IPlugin(ABC):
    """Plugin interface."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup plugin."""
        pass


class PluginManager:
    """Plugin manager."""
    
    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        plugin: IPlugin,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register plugin."""
        metadata = plugin.get_metadata()
        
        if metadata.name in self._plugins:
            logger.warning(f"Plugin {metadata.name} already registered, replacing...")
        
        self._plugins[metadata.name] = plugin
        self._metadata[metadata.name] = metadata
        self._configs[metadata.name] = config or {}
        
        if metadata.enabled:
            plugin.initialize(self._configs[metadata.name])
        
        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
    
    def load_from_module(self, module_path: str, plugin_class: str):
        """Load plugin from module."""
        try:
            module = importlib.import_module(module_path)
            plugin_class_obj = getattr(module, plugin_class)
            plugin = plugin_class_obj()
            self.register(plugin)
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")
    
    def enable(self, plugin_name: str):
        """Enable plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].initialize(self._configs[plugin_name])
            self._metadata[plugin_name].enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")
    
    def disable(self, plugin_name: str):
        """Disable plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].cleanup()
            self._metadata[plugin_name].enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
    
    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all plugins."""
        return list(self._metadata.values())
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        if not self._metadata[plugin_name].enabled:
            raise ValueError(f"Plugin {plugin_name} is disabled")
        
        return plugin.execute(*args, **kwargs)
    
    def cleanup_all(self):
        """Cleanup all plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin: {e}")















