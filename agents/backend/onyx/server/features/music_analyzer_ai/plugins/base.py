"""
Base Plugin System
Foundation for extensible plugin architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled
        }


class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
    
    def register(self, plugin: BasePlugin):
        """Register a plugin"""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered, overwriting")
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
    
    def unregister(self, name: str):
        """Unregister a plugin"""
        if name in self._plugins:
            self._plugins[name].cleanup()
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
    
    def get(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name"""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self._plugins.keys())
    
    def get_all(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins"""
        return self._plugins.copy()



