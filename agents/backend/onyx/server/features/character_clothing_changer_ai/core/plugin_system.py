"""
Plugin System
=============

Extensible plugin system for custom enhancements.
"""

import logging
import importlib
import importlib.util
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EnhancementPlugin(ABC):
    """
    Base class for enhancement plugins.
    
    Plugins can extend functionality by:
    - Adding custom enhancement types
    - Modifying processing pipelines
    - Adding post-processing steps
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
    
    @abstractmethod
    def process(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process enhancement.
        
        Args:
            file_path: Path to file
            parameters: Processing parameters
            
        Returns:
            Result dictionary
        """
        pass
    
    def validate(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if valid
        """
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for plugin.
        
        Returns:
            JSON schema dictionary
        """
        return {}


class PluginManager:
    """
    Manages enhancement plugins.
    
    Features:
    - Plugin registration
    - Plugin discovery
    - Plugin execution
    - Plugin validation
    """
    
    def __init__(self):
        """Initialize plugin manager."""
        self._plugins: Dict[str, EnhancementPlugin] = {}
        self._enabled_plugins: List[str] = []
    
    def register(self, plugin: EnhancementPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
        """
        name = plugin.get_name()
        self._plugins[name] = plugin
        logger.info(f"Registered plugin: {name} v{plugin.get_version()}")
    
    def unregister(self, plugin_name: str):
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to remove
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            if plugin_name in self._enabled_plugins:
                self._enabled_plugins.remove(plugin_name)
            logger.info(f"Unregistered plugin: {plugin_name}")
    
    def enable(self, plugin_name: str):
        """Enable a plugin."""
        if plugin_name in self._plugins:
            if plugin_name not in self._enabled_plugins:
                self._enabled_plugins.append(plugin_name)
                logger.info(f"Enabled plugin: {plugin_name}")
        else:
            logger.warning(f"Plugin not found: {plugin_name}")
    
    def disable(self, plugin_name: str):
        """Disable a plugin."""
        if plugin_name in self._enabled_plugins:
            self._enabled_plugins.remove(plugin_name)
            logger.info(f"Disabled plugin: {plugin_name}")
    
    def get_plugin(self, plugin_name: str) -> Optional[EnhancementPlugin]:
        """Get plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                "name": name,
                "version": plugin.get_version(),
                "enabled": name in self._enabled_plugins,
                "config_schema": plugin.get_config_schema()
            }
            for name, plugin in self._plugins.items()
        ]
    
    async def execute_plugin(
        self,
        plugin_name: str,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a plugin.
        
        Args:
            plugin_name: Name of plugin
            file_path: Path to file
            parameters: Processing parameters
            
        Returns:
            Result dictionary
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if plugin_name not in self._enabled_plugins:
            raise ValueError(f"Plugin not enabled: {plugin_name}")
        
        # Validate parameters
        if not plugin.validate(parameters):
            raise ValueError(f"Invalid parameters for plugin {plugin_name}")
        
        # Execute plugin
        try:
            result = plugin.process(file_path, parameters)
            return {
                "success": True,
                "plugin": plugin_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing plugin {plugin_name}: {e}", exc_info=True)
            return {
                "success": False,
                "plugin": plugin_name,
                "error": str(e)
            }
    
    def load_from_directory(self, directory: str):
        """
        Load plugins from directory.
        
        Args:
            directory: Directory containing plugin modules
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
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and
                            issubclass(attr, EnhancementPlugin) and
                            attr != EnhancementPlugin):
                            plugin = attr()
                            self.register(plugin)
                            logger.info(f"Loaded plugin from {plugin_file}: {plugin.get_name()}")
            except Exception as e:
                logger.error(f"Error loading plugin from {plugin_file}: {e}", exc_info=True)

