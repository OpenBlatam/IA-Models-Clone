"""
Plugin System for Piel Mejorador AI SAM3
========================================

Extensible plugin architecture.
"""

import importlib
import importlib.util
import logging
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Interface for plugins."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin."""
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data."""
        pass
    
    def cleanup(self):
        """Cleanup plugin resources."""
        pass


@dataclass
class PluginInfo:
    """Plugin information."""
    name: str
    version: str
    enabled: bool
    path: Optional[Path] = None


class PluginManager:
    """
    Manages plugins for extensibility.
    
    Features:
    - Dynamic plugin loading
    - Plugin lifecycle management
    - Plugin configuration
    - Plugin dependencies
    """
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = plugin_dir or Path("plugins")
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, plugin: PluginInterface, config: Optional[Dict[str, Any]] = None):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
            config: Optional plugin configuration
        """
        name = plugin.get_name()
        
        if plugin.initialize(config or {}):
            self._plugins[name] = plugin
            self._plugin_info[name] = PluginInfo(
                name=name,
                version=plugin.get_version(),
                enabled=True
            )
            self._configs[name] = config or {}
            logger.info(f"Plugin registered: {name} v{plugin.get_version()}")
        else:
            logger.error(f"Failed to initialize plugin: {name}")
    
    def load_plugins_from_directory(self):
        """Load plugins from plugin directory."""
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            try:
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for Plugin class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, PluginInterface) and
                        attr != PluginInterface):
                        plugin = attr()
                        self.register_plugin(plugin)
            except Exception as e:
                logger.error(f"Error loading plugin from {plugin_file}: {e}")
    
    def process_with_plugins(
        self,
        data: Dict[str, Any],
        plugin_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process data through plugins.
        
        Args:
            data: Data to process
            plugin_names: Optional list of plugin names (all if None)
            
        Returns:
            Processed data
        """
        plugins_to_use = plugin_names or list(self._plugins.keys())
        
        result = data
        for plugin_name in plugins_to_use:
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                try:
                    result = plugin.process(result)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin_name}: {e}")
        
        return result
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugin_info.values())
    
    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self._plugin_info:
            self._plugin_info[name].enabled = True
            return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self._plugin_info:
            self._plugin_info[name].enabled = False
            return True
        return False
    
    def unload_plugin(self, name: str):
        """Unload a plugin."""
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.cleanup()
            del self._plugins[name]
            del self._plugin_info[name]
            logger.info(f"Plugin unloaded: {name}")
    
    def cleanup_all(self):
        """Cleanup all plugins."""
        for plugin in self._plugins.values():
            plugin.cleanup()
        self._plugins.clear()
        self._plugin_info.clear()

