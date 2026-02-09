"""
Base Plugin Interface
=====================

Base classes for plugin system.
Allows extending functionality through plugins.

Author: BUL System
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """
    Base class for all plugins.
    
    Plugins allow extending trainer functionality without modifying core code.
    
    Attributes:
        name: Plugin name
        version: Plugin version
        enabled: Whether plugin is enabled
        
    Example:
        >>> class MyPlugin(BasePlugin):
        ...     def __init__(self):
        ...         super().__init__("my_plugin", "1.0.0")
        ...     
        ...     def on_train_begin(self, trainer, **kwargs):
        ...         print("Training started!")
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
        """
        self.name = name
        self.version = version
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    def configure(self, **kwargs) -> None:
        """
        Configure plugin with options.
        
        Args:
            **kwargs: Configuration options
        """
        self.config.update(kwargs)
        logger.debug(f"Plugin {self.name} configured with {kwargs}")
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
        logger.debug(f"Plugin {self.name} enabled")
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
        logger.debug(f"Plugin {self.name} disabled")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.
        
        Returns:
            Dictionary with plugin info
        """
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "config": self.config,
        }


class PluginRegistry:
    """
    Registry for managing plugins.
    
    Allows registration, discovery, and management of plugins.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(MyPlugin())
        >>> plugins = registry.get_all()
    """
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, BasePlugin] = {}
    
    def register(self, plugin: BasePlugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered, overwriting")
        self._plugins[plugin.name] = plugin
        logger.info(f"Plugin {plugin.name} v{plugin.version} registered")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Plugin {name} unregistered")
    
    def get(self, name: str) -> Optional[BasePlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def get_all(self) -> List[BasePlugin]:
        """
        Get all registered plugins.
        
        Returns:
            List of all plugins
        """
        return list(self._plugins.values())
    
    def get_enabled(self) -> List[BasePlugin]:
        """
        Get all enabled plugins.
        
        Returns:
            List of enabled plugins
        """
        return [p for p in self._plugins.values() if p.enabled]
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        logger.info("Plugin registry cleared")

