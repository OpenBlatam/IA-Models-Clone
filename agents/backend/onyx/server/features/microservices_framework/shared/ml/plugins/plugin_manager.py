"""
Plugin Manager
Plugin system for extending framework functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base plugin class."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup plugin resources."""
        pass


class ModelPlugin(Plugin):
    """Plugin for custom model architectures."""
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create a model instance."""
        pass


class DataPlugin(Plugin):
    """Plugin for custom data processing."""
    
    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> Any:
        """Process data."""
        pass


class TrainingPlugin(Plugin):
    """Plugin for custom training strategies."""
    
    @abstractmethod
    def train_step(self, model: Any, batch: Any, **kwargs) -> Dict[str, float]:
        """Perform a training step."""
        pass


class PluginManager:
    """Manager for plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, plugin: Plugin, config: Optional[Dict[str, Any]] = None):
        """Register a plugin."""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered, overwriting")
        
        self._plugins[plugin.name] = plugin
        if config:
            self._plugin_configs[plugin.name] = config
        
        # Initialize plugin
        try:
            plugin.initialize(config)
            logger.info(f"Plugin {plugin.name} v{plugin.version} registered and initialized")
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin.name}: {e}")
            raise
    
    def unregister(self, plugin_name: str):
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.cleanup()
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_configs:
                del self._plugin_configs[plugin_name]
            logger.info(f"Plugin {plugin_name} unregistered")
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def load_plugin_from_module(self, module_path: str, plugin_class_name: str):
        """Load plugin from a Python module."""
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)
            plugin = plugin_class()
            self.register(plugin)
            return plugin
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")
            raise
    
    def load_plugins_from_directory(self, directory: str):
        """Load all plugins from a directory."""
        plugin_dir = Path(directory)
        
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                module_name = f"{plugin_dir.name}.{plugin_file.stem}"
                # This is simplified - in practice, you'd need proper module loading
                logger.info(f"Loading plugin from {plugin_file}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
    
    def cleanup_all(self):
        """Cleanup all plugins."""
        for plugin_name in list(self._plugins.keys()):
            self.unregister(plugin_name)



