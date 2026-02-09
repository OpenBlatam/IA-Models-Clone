"""
Plugin Base Classes

Base classes for creating extensible plugins.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from dataclasses import dataclass


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    dependencies: Optional[list[str]] = None


class BasePlugin(ABC):
    """
    Base class for all plugins.
    
    Plugins extend functionality without modifying core code.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or {}
        self._enabled = True
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable plugin."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable plugin."""
        self._enabled = False
    
    def initialize(self) -> None:
        """Initialize plugin (called when plugin is registered)."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin (called when plugin is unregistered)."""
        pass


class PluginRegistry:
    """
    Registry for managing plugins.
    """
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, BasePlugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
    
    def register(self, plugin: BasePlugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        metadata = plugin.metadata
        name = metadata.name
        
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        
        # Check dependencies
        if metadata.dependencies:
            for dep in metadata.dependencies:
                if dep not in self._plugins:
                    raise ValueError(f"Plugin '{name}' requires dependency '{dep}' which is not registered")
        
        self._plugins[name] = plugin
        self._metadata[name] = metadata
        
        # Initialize plugin
        plugin.initialize()
    
    def unregister(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered")
        
        plugin = self._plugins[name]
        plugin.cleanup()
        
        del self._plugins[name]
        del self._metadata[name]
    
    def get(self, name: str) -> Optional[BasePlugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def get_all(self) -> Dict[str, BasePlugin]:
        """
        Get all registered plugins.
        
        Returns:
            Dictionary of plugin name to plugin instance
        """
        return self._plugins.copy()
    
    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin metadata or None
        """
        return self._metadata.get(name)
    
    def list_plugins(self) -> list[str]:
        """
        List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if plugin is registered.
        
        Args:
            name: Plugin name
        
        Returns:
            True if registered
        """
        return name in self._plugins



