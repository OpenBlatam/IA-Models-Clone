"""
Plugin Manager
Manages plugin lifecycle and execution
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import importlib.util

from .base import BasePlugin, PluginRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """Manager for plugin system"""
    
    def __init__(self, plugin_dir: Optional[str] = None):
        self.registry = PluginRegistry()
        self.plugin_dir = Path(plugin_dir) if plugin_dir else Path("./plugins")
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
    
    def register_plugin(self, plugin: BasePlugin, config: Optional[Dict[str, Any]] = None):
        """Register and initialize a plugin"""
        if plugin.initialize(config):
            self.registry.register(plugin)
        else:
            logger.error(f"Failed to initialize plugin: {plugin.name}")
    
    def load_plugin_from_file(self, file_path: str) -> bool:
        """Load plugin from Python file"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return False
        
        try:
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for Plugin class
            if hasattr(module, 'Plugin'):
                plugin_class = module.Plugin
                if issubclass(plugin_class, BasePlugin):
                    plugin = plugin_class()
                    self.register_plugin(plugin)
                    return True
            
            logger.error(f"No valid Plugin class found in {file_path}")
            return False
        
        except Exception as e:
            logger.error(f"Error loading plugin from {file_path}: {str(e)}")
            return False
    
    def load_plugins_from_directory(self, directory: Optional[str] = None):
        """Load all plugins from directory"""
        directory = Path(directory) if directory else self.plugin_dir
        
        if not directory.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return
        
        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            self.load_plugin_from_file(plugin_file)
    
    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """Execute a plugin"""
        plugin = self.registry.get(name)
        if plugin is None:
            logger.error(f"Plugin not found: {name}")
            return None
        
        if not plugin.enabled:
            logger.warning(f"Plugin {name} is disabled")
            return None
        
        try:
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin {name}: {str(e)}")
            return None
    
    def enable_plugin(self, name: str):
        """Enable a plugin"""
        plugin = self.registry.get(name)
        if plugin:
            plugin.enabled = True
            logger.info(f"Enabled plugin: {name}")
    
    def disable_plugin(self, name: str):
        """Disable a plugin"""
        plugin = self.registry.get(name)
        if plugin:
            plugin.enabled = False
            logger.info(f"Disabled plugin: {name}")
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return self.registry.list_plugins()
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information"""
        plugin = self.registry.get(name)
        return plugin.get_info() if plugin else None



