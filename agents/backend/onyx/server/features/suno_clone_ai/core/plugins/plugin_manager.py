"""
Plugin Manager

Utilities for plugin management and loading.
"""

import logging
import importlib
import importlib.util
from typing import Dict, Any, Optional, Type
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginManager:
    """Manage plugins."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Any] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        plugin_name: str,
        plugin_class: Type,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register plugin.
        
        Args:
            plugin_name: Plugin name
            plugin_class: Plugin class
            metadata: Optional metadata
        """
        self.plugins[plugin_name] = plugin_class
        self.plugin_metadata[plugin_name] = metadata or {}
        logger.info(f"Registered plugin: {plugin_name}")
    
    def load(
        self,
        plugin_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Load plugin instance.
        
        Args:
            plugin_name: Plugin name
            *args: Plugin constructor arguments
            **kwargs: Plugin constructor keyword arguments
            
        Returns:
            Plugin instance
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        plugin_class = self.plugins[plugin_name]
        instance = plugin_class(*args, **kwargs)
        
        logger.info(f"Loaded plugin: {plugin_name}")
        
        return instance
    
    def load_from_module(
        self,
        module_path: str,
        plugin_name: str
    ) -> None:
        """
        Load plugin from module.
        
        Args:
            module_path: Module path
            plugin_name: Plugin name
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_name)
            self.register(plugin_name, plugin_class)
        except Exception as e:
            logger.error(f"Error loading plugin: {e}")
            raise
    
    def load_from_file(
        self,
        file_path: str,
        plugin_name: str
    ) -> None:
        """
        Load plugin from file.
        
        Args:
            file_path: Plugin file path
            plugin_name: Plugin name
        """
        # Load plugin from Python file
        spec = importlib.util.spec_from_file_location(plugin_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        plugin_class = getattr(module, plugin_name)
        self.register(plugin_name, plugin_class)
    
    def get_plugin(self, plugin_name: str) -> Optional[Type]:
        """
        Get plugin class.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin class or None
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> list:
        """List all registered plugins."""
        return list(self.plugins.keys())


def load_plugin(
    manager: PluginManager,
    plugin_name: str,
    **kwargs
) -> Any:
    """Load plugin."""
    return manager.load(plugin_name, **kwargs)


def register_plugin(
    manager: PluginManager,
    plugin_name: str,
    plugin_class: Type,
    **kwargs
) -> None:
    """Register plugin."""
    manager.register(plugin_name, plugin_class, **kwargs)


def get_plugin(
    manager: PluginManager,
    plugin_name: str
) -> Optional[Type]:
    """Get plugin."""
    return manager.get_plugin(plugin_name)
