"""
Plugin Manager for Color Grading AI
====================================

Manages plugins for extending functionality.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from abc import ABC, abstractmethod
import inspect

logger = logging.getLogger(__name__)


class ColorGradingPlugin(ABC):
    """Base class for color grading plugins."""
    
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
    async def process(self, media_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process media with plugin.
        
        Args:
            media_path: Path to media file
            params: Processing parameters
            
        Returns:
            Processing result
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate plugin parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if valid
        """
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": getattr(self, "description", ""),
        }


class PluginManager:
    """
    Manages color grading plugins.
    
    Features:
    - Load plugins dynamically
    - Register/unregister plugins
    - Execute plugins
    - Plugin discovery
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        """
        Initialize plugin manager.
        
        Args:
            plugins_dir: Directory containing plugins
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: Dict[str, ColorGradingPlugin] = {}
        self._load_plugins()
    
    def _load_plugins(self):
        """Load plugins from directory."""
        # Look for Python files in plugins directory
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                self._load_plugin_file(plugin_file)
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file}: {e}")
    
    def _load_plugin_file(self, plugin_file: Path):
        """Load plugin from file."""
        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ColorGradingPlugin) and 
                    obj != ColorGradingPlugin):
                    plugin = obj()
                    self.register(plugin)
                    logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
    
    def register(self, plugin: ColorGradingPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
        """
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")
    
    def unregister(self, plugin_name: str):
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Plugin name
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
    
    def get_plugin(self, plugin_name: str) -> Optional[ColorGradingPlugin]:
        """
        Get plugin by name.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin information
        """
        return [plugin.get_info() for plugin in self._plugins.values()]
    
    async def execute_plugin(
        self,
        plugin_name: str,
        media_path: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a plugin.
        
        Args:
            plugin_name: Plugin name
            media_path: Path to media file
            params: Processing parameters
            
        Returns:
            Processing result
            
        Raises:
            ValueError: If plugin not found
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Validate parameters
        if not plugin.validate_params(params):
            raise ValueError(f"Invalid parameters for plugin {plugin_name}")
        
        # Execute plugin
        return await plugin.process(media_path, params)


# Import for plugin file loading
import importlib.util
import importlib

