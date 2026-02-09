"""
Plugins Module

Plugin system for extending generator functionality.
"""

from typing import Dict, Any, List, Optional, Callable, Type
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class GeneratorPlugin(ABC):
    """
    Base class for generator plugins.
    """
    
    @abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def version(self) -> str:
        """Get plugin version."""
        pass
    
    def before_create(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before generator creation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Modified configuration
        """
        return config
    
    def after_create(self, generator: Any, config: Dict[str, Any]) -> Any:
        """
        Called after generator creation.
        
        Args:
            generator: Created generator instance
            config: Configuration used
            
        Returns:
            Modified generator (or original)
        """
        return generator
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            (is_valid, error_message)
        """
        return True, None


class PluginManager:
    """
    Manages generator plugins.
    """
    
    def __init__(self):
        self._plugins: Dict[str, GeneratorPlugin] = {}
    
    def register(self, plugin: GeneratorPlugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin to register
        """
        name = plugin.name()
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered, overwriting")
        self._plugins[name] = plugin
        logger.info(f"Plugin {name} v{plugin.version()} registered")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Plugin {name} unregistered")
    
    def get_plugin(self, name: str) -> Optional[GeneratorPlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def apply_before_create(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all plugins' before_create hooks.
        
        Args:
            config: Configuration
            
        Returns:
            Modified configuration
        """
        result = config.copy()
        for plugin in self._plugins.values():
            try:
                result = plugin.before_create(result)
            except Exception as e:
                logger.error(f"Error in plugin {plugin.name()} before_create: {e}")
        return result
    
    def apply_after_create(self, generator: Any, config: Dict[str, Any]) -> Any:
        """
        Apply all plugins' after_create hooks.
        
        Args:
            generator: Generator instance
            config: Configuration
            
        Returns:
            Modified generator
        """
        result = generator
        for plugin in self._plugins.values():
            try:
                result = plugin.after_create(result, config)
            except Exception as e:
                logger.error(f"Error in plugin {plugin.name()} after_create: {e}")
        return result
    
    def validate_with_plugins(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate configuration with all plugins.
        
        Args:
            config: Configuration to validate
            
        Returns:
            (is_valid, error_message)
        """
        for plugin in self._plugins.values():
            is_valid, error = plugin.validate_config(config)
            if not is_valid:
                return False, f"Plugin {plugin.name()}: {error}"
        return True, None


# Global plugin manager
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    return _plugin_manager


def register_plugin(plugin: GeneratorPlugin) -> None:
    """Register a plugin."""
    _plugin_manager.register(plugin)


def unregister_plugin(name: str) -> None:
    """Unregister a plugin."""
    _plugin_manager.unregister(name)















