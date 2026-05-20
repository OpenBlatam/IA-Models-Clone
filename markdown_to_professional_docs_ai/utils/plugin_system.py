"""Plugin system for extensibility"""
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import importlib
import importlib.util
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """Base class for plugins"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin"""
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data"""
        pass


class PluginManager:
    """Manage plugins"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        """
        Initialize plugin manager
        
        Args:
            plugins_dir: Directory containing plugins
        """
        if plugins_dir is None:
            plugins_dir = Path(__file__).parent.parent / "plugins"
        
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: Dict[str, BasePlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Register a plugin
        
        Args:
            plugin: Plugin instance
        """
        name = plugin.get_name()
        self._plugins[name] = plugin
        logger.info(f"Plugin registered: {name} v{plugin.get_version()}")
    
    def load_plugin(self, plugin_path: str) -> Optional[BasePlugin]:
        """
        Load plugin from file
        
        Args:
            plugin_path: Path to plugin file
            
        Returns:
            Plugin instance or None
        """
        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin = attr()
                    self.register_plugin(plugin)
                    return plugin
            
            return None
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {e}")
            return None
    
    def load_plugins_from_directory(self) -> List[str]:
        """
        Load all plugins from plugins directory
        
        Returns:
            List of loaded plugin names
        """
        loaded = []
        
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            plugin = self.load_plugin(str(plugin_file))
            if plugin:
                loaded.append(plugin.get_name())
        
        return loaded
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name"""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, str]]:
        """List all registered plugins"""
        return [
            {
                "name": plugin.get_name(),
                "version": plugin.get_version()
            }
            for plugin in self._plugins.values()
        ]
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback
        
        Args:
            hook_name: Hook name
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hook callbacks
        
        Args:
            hook_name: Hook name
            data: Data to pass to callbacks
            
        Returns:
            Modified data
        """
        if hook_name not in self._hooks:
            return data
        
        result = data
        for callback in self._hooks[hook_name]:
            try:
                result = callback(result)
            except Exception as e:
                logger.error(f"Error executing hook {hook_name}: {e}")
        
        return result


# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

