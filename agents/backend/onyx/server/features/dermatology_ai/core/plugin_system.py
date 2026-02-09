"""
Plugin System for Modular Architecture
Allows dynamic loading and registration of plugins
"""

import importlib
import inspect
from typing import Dict, List, Optional, Type, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Plugin types"""
    MIDDLEWARE = "middleware"
    ROUTER = "router"
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_BROKER = "message_broker"
    AUTHENTICATION = "authentication"
    OBSERVABILITY = "observability"


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: Optional[str] = None
    dependencies: List[str] = None
    config_schema: Optional[Dict[str, Any]] = None


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Shutdown plugin"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration"""
        if self.metadata.config_schema:
            # Basic validation - can be extended with JSON Schema
            required_fields = self.metadata.config_schema.get("required", [])
            for field in required_fields:
                if field not in config:
                    return False
        return True


class PluginRegistry:
    """
    Central registry for plugins.
    Supports dynamic loading and registration.
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self.initialized: Dict[str, bool] = {}
    
    def register(self, plugin: BasePlugin):
        """Register a plugin"""
        metadata = plugin.metadata
        plugin_id = f"{metadata.plugin_type.value}:{metadata.name}"
        
        if plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin_id} already registered, overwriting")
        
        self.plugins[plugin_id] = plugin
        self.plugins_by_type[metadata.plugin_type].append(plugin_id)
        self.initialized[plugin_id] = False
        
        logger.info(f"Registered plugin: {plugin_id} v{metadata.version}")
    
    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Get plugin by ID"""
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type"""
        plugin_ids = self.plugins_by_type.get(plugin_type, [])
        return [self.plugins[pid] for pid in plugin_ids if pid in self.plugins]
    
    async def initialize_plugin(
        self,
        plugin_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Initialize a plugin"""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            logger.error(f"Plugin {plugin_id} not found")
            return False
        
        if self.initialized.get(plugin_id, False):
            logger.warning(f"Plugin {plugin_id} already initialized")
            return True
        
        try:
            # Validate config
            if config and not plugin.validate_config(config):
                logger.error(f"Invalid config for plugin {plugin_id}")
                return False
            
            await plugin.initialize(config)
            self.initialized[plugin_id] = True
            logger.info(f"Initialized plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_id}: {e}", exc_info=True)
            return False
    
    async def initialize_all(
        self,
        plugin_type: Optional[PluginType] = None,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize all plugins or plugins of a specific type"""
        configs = configs or {}
        
        if plugin_type:
            plugins = self.get_plugins_by_type(plugin_type)
        else:
            plugins = list(self.plugins.values())
        
        for plugin in plugins:
            plugin_id = f"{plugin.metadata.plugin_type.value}:{plugin.metadata.name}"
            config = configs.get(plugin_id, {})
            await self.initialize_plugin(plugin_id, config)
    
    async def shutdown_all(self):
        """Shutdown all initialized plugins"""
        for plugin_id, plugin in self.plugins.items():
            if self.initialized.get(plugin_id, False):
                try:
                    await plugin.shutdown()
                    self.initialized[plugin_id] = False
                    logger.info(f"Shutdown plugin: {plugin_id}")
                except Exception as e:
                    logger.error(f"Error shutting down plugin {plugin_id}: {e}")
    
    def load_from_module(self, module_path: str):
        """Load plugins from a Python module"""
        try:
            module = importlib.import_module(module_path)
            
            # Find all BasePlugin subclasses
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and 
                    obj.__module__ == module_path):
                    plugin_instance = obj()
                    self.register(plugin_instance)
                    logger.info(f"Loaded plugin from {module_path}: {name}")
                    
        except Exception as e:
            logger.error(f"Failed to load plugins from {module_path}: {e}", exc_info=True)
    
    def load_from_directory(self, directory: str):
        """Load plugins from a directory"""
        import os
        from pathlib import Path
        
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return
        
        # Find all Python files
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            module_name = py_file.stem
            module_path = f"{directory.replace('/', '.')}.{module_name}"
            
            try:
                self.load_from_module(module_path)
            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")


# Global plugin registry
_plugin_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create global plugin registry"""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry















