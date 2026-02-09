"""
Plugin system for extending functionality.
"""

from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plugin logic.
        
        Args:
            data: Input data
        
        Returns:
            Output data
        """
        pass
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class PluginManager:
    """Manages plugins for the system."""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, plugin: Plugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
        """
        self.plugins[plugin.name] = plugin
        logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")
    
    def unregister_plugin(self, plugin_name: str):
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            del self.plugins[plugin_name]
            logger.info(f"Plugin unregistered: {plugin_name}")
    
    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
        logger.info(f"Hook registered: {hook_name}")
    
    def execute_hook(self, hook_name: str, data: Any = None) -> List[Any]:
        """
        Execute all callbacks for a hook.
        
        Args:
            hook_name: Name of the hook
            data: Data to pass to callbacks
        
        Returns:
            List of results from callbacks
        """
        results = []
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(data) if data is not None else callback()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error executing hook {hook_name}: {e}")
        return results
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of plugin
        
        Returns:
            Plugin instance or None
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin information
        """
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "enabled": plugin.enabled
            }
            for plugin in self.plugins.values()
        ]






