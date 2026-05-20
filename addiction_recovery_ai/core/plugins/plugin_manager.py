"""
Plugin Manager
System for extending functionality with plugins
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


class Plugin:
    """
    Base plugin class
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize plugin
        
        Args:
            name: Plugin name
            version: Plugin version
        """
        self.name = name
        self.version = version
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin
        
        Args:
            config: Plugin configuration
        """
        pass
    
    def cleanup(self):
        """Cleanup plugin resources"""
        pass


class PluginManager:
    """
    Manager for loading and managing plugins
    """
    
    def __init__(self):
        """Initialize plugin manager"""
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, plugin: Plugin):
        """
        Register a plugin
        
        Args:
            plugin: Plugin instance
        """
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
    
    def load_plugin(self, plugin_path: str, plugin_name: str):
        """
        Load plugin from file
        
        Args:
            plugin_path: Path to plugin module
            plugin_name: Name of plugin class
        """
        try:
            module = importlib.import_module(plugin_path)
            plugin_class = getattr(module, plugin_name)
            plugin = plugin_class()
            self.register_plugin(plugin)
            return plugin
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise
    
    def load_plugins_from_directory(self, directory: str):
        """
        Load all plugins from directory
        
        Args:
            directory: Directory containing plugins
        """
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                module_name = plugin_file.stem
                module = importlib.import_module(f"{directory}.{module_name}")
                
                # Look for Plugin subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Plugin) and 
                        attr != Plugin):
                        plugin = attr()
                        self.register_plugin(plugin)
            except Exception as e:
                logger.warning(f"Failed to load plugin from {plugin_file}: {e}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get plugin by name
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self.plugins.keys())
    
    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register a hook callback
        
        Args:
            hook_name: Name of hook
            callback: Callback function
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Call all registered hooks
        
        Args:
            hook_name: Name of hook
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of hook return values
        """
        results = []
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook {hook_name} failed: {e}")
        return results
    
    def unregister_plugin(self, name: str):
        """
        Unregister plugin
        
        Args:
            name: Plugin name
        """
        if name in self.plugins:
            plugin = self.plugins[name]
            plugin.cleanup()
            del self.plugins[name]
            logger.info(f"Unregistered plugin: {name}")
    
    def cleanup_all(self):
        """Cleanup all plugins"""
        for plugin in self.plugins.values():
            plugin.cleanup()
        self.plugins.clear()
        self.hooks.clear()


# Global plugin manager instance
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance"""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager













