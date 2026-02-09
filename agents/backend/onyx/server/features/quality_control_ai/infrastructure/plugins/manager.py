"""
Plugin Manager

Manages plugin lifecycle and execution.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from .base import BasePlugin, PluginRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manager for plugin lifecycle and hooks.
    """
    
    def __init__(self):
        """Initialize plugin manager."""
        self.registry = PluginRegistry()
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin to register
        """
        try:
            self.registry.register(plugin)
            logger.info(f"Plugin '{plugin.metadata.name}' registered successfully")
        except Exception as e:
            logger.error(f"Failed to register plugin '{plugin.metadata.name}': {e}")
            raise
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        try:
            self.registry.unregister(name)
            logger.info(f"Plugin '{name}' unregistered successfully")
        except Exception as e:
            logger.error(f"Failed to unregister plugin '{name}': {e}")
            raise
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Execute all callbacks for a hook.
        
        Args:
            hook_name: Name of the hook
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            List of return values from callbacks
        """
        results = []
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error executing hook '{hook_name}': {e}")
        return results
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None
        """
        return self.registry.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin names
        """
        return self.registry.list_plugins()
    
    def enable_plugin(self, name: str) -> None:
        """
        Enable a plugin.
        
        Args:
            name: Plugin name
        """
        plugin = self.registry.get(name)
        if plugin:
            plugin.enable()
            logger.info(f"Plugin '{name}' enabled")
        else:
            raise ValueError(f"Plugin '{name}' not found")
    
    def disable_plugin(self, name: str) -> None:
        """
        Disable a plugin.
        
        Args:
            name: Plugin name
        """
        plugin = self.registry.get(name)
        if plugin:
            plugin.disable()
            logger.info(f"Plugin '{name}' disabled")
        else:
            raise ValueError(f"Plugin '{name}' not found")



