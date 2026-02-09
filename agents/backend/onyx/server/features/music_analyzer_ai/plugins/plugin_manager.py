"""
Plugin Manager - Manages plugin lifecycle
"""

from typing import Dict, Any, Optional, List, Type
import logging
import importlib
import importlib.util
from pathlib import Path

from ..interfaces.plugin_interface import IPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages plugin registration and execution
    """
    
    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, plugin: IPlugin, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a plugin
        
        Args:
            plugin: Plugin instance
            config: Plugin configuration
        """
        plugin_name = plugin.name
        self._plugins[plugin_name] = plugin
        self._plugin_configs[plugin_name] = config or {}
        
        # Initialize plugin
        try:
            plugin.initialize(self._plugin_configs[plugin_name])
            logger.info(f"Registered and initialized plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Error initializing plugin {plugin_name}: {str(e)}")
    
    def unregister(self, plugin_name: str) -> None:
        """Unregister a plugin"""
        if plugin_name in self._plugins:
            try:
                self._plugins[plugin_name].cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin_name}: {str(e)}")
            
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_configs:
                del self._plugin_configs[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
    
    def get(self, plugin_name: str) -> Optional[IPlugin]:
        """Get plugin by name"""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self._plugins.keys())
    
    def execute_plugin(self, plugin_name: str, data: Any) -> Any:
        """Execute a plugin"""
        plugin = self.get(plugin_name)
        if plugin is None:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        return plugin.execute(data)
    
    def load_from_directory(self, directory: str) -> None:
        """Load plugins from directory"""
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, IPlugin) and 
                        attr != IPlugin):
                        plugin_instance = attr()
                        self.register(plugin_instance)
                        logger.info(f"Loaded plugin from {plugin_file}: {attr_name}")
            
            except Exception as e:
                logger.error(f"Error loading plugin from {plugin_file}: {str(e)}")


# Global plugin manager
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager"""
    return _plugin_manager


def register_plugin(plugin: IPlugin, config: Optional[Dict[str, Any]] = None) -> None:
    """Register plugin in global manager"""
    _plugin_manager.register(plugin, config)


def get_plugin(plugin_name: str) -> Optional[IPlugin]:
    """Get plugin from global manager"""
    return _plugin_manager.get(plugin_name)

