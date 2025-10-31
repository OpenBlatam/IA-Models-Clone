"""
Plugin Manager for Export IA
============================

Advanced plugin management system for dynamic loading, discovery,
and management of Export IA components and extensions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Type, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import sys
import importlib
import pkgutil
import inspect
from pathlib import Path
import zipfile
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml
import jsonschema

from .base_component import (
    BaseComponent, ComponentRegistry, ComponentMetadata, ComponentConfig,
    ComponentPriority, get_global_registry, get_global_event_bus
)

logger = logging.getLogger(__name__)

class PluginStatus(Enum):
    """Plugin status states."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UNINSTALLED = "uninstalled"

class PluginType(Enum):
    """Types of plugins."""
    COMPONENT = "component"
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    EXPORTER = "exporter"
    INTEGRATION = "integration"
    THEME = "theme"
    EXTENSION = "extension"

@dataclass
class PluginInfo:
    """Plugin information and metadata."""
    id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    entry_point: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_system_version: str = "1.0.0"
    max_system_version: str = "*"
    license: str = "MIT"
    repository: str = ""
    documentation: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PluginManifest:
    """Plugin manifest file structure."""
    plugin_info: PluginInfo
    components: List[Dict[str, Any]] = field(default_factory=list)
    hooks: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    migrations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PluginInstance:
    """Plugin instance with runtime information."""
    plugin_info: PluginInfo
    status: PluginStatus
    module: Optional[Any] = None
    components: List[BaseComponent] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    loaded_at: Optional[datetime] = None
    enabled_at: Optional[datetime] = None

class PluginDiscovery:
    """Plugin discovery and scanning system."""
    
    def __init__(self, plugin_directories: List[str]):
        self.plugin_directories = plugin_directories
        self.discovered_plugins: Dict[str, PluginInfo] = {}
        self._lock = threading.RLock()
    
    async def discover_plugins(self) -> Dict[str, PluginInfo]:
        """Discover all available plugins."""
        with self._lock:
            self.discovered_plugins.clear()
            
            for directory in self.plugin_directories:
                await self._scan_directory(directory)
            
            logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
            return self.discovered_plugins.copy()
    
    async def _scan_directory(self, directory: str) -> None:
        """Scan a directory for plugins."""
        try:
            path = Path(directory)
            if not path.exists():
                logger.warning(f"Plugin directory does not exist: {directory}")
                return
            
            # Scan for plugin packages
            for item in path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    await self._scan_plugin_directory(item)
                elif item.suffix == '.zip':
                    await self._scan_plugin_archive(item)
        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    async def _scan_plugin_directory(self, plugin_path: Path) -> None:
        """Scan a plugin directory."""
        try:
            manifest_path = plugin_path / "plugin.yaml"
            if not manifest_path.exists():
                manifest_path = plugin_path / "plugin.yml"
            
            if manifest_path.exists():
                plugin_info = await self._load_plugin_manifest(manifest_path)
                if plugin_info:
                    self.discovered_plugins[plugin_info.id] = plugin_info
            else:
                # Try to auto-discover plugin info
                plugin_info = await self._auto_discover_plugin(plugin_path)
                if plugin_info:
                    self.discovered_plugins[plugin_info.id] = plugin_info
        
        except Exception as e:
            logger.error(f"Error scanning plugin directory {plugin_path}: {e}")
    
    async def _scan_plugin_archive(self, archive_path: Path) -> None:
        """Scan a plugin archive file."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                
                temp_path = Path(temp_dir)
                await self._scan_plugin_directory(temp_path)
        
        except Exception as e:
            logger.error(f"Error scanning plugin archive {archive_path}: {e}")
    
    async def _load_plugin_manifest(self, manifest_path: Path) -> Optional[PluginInfo]:
        """Load plugin manifest file."""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = yaml.safe_load(f)
            
            # Validate manifest schema
            if not self._validate_manifest(manifest_data):
                logger.error(f"Invalid manifest file: {manifest_path}")
                return None
            
            # Extract plugin info
            plugin_data = manifest_data.get('plugin', {})
            plugin_info = PluginInfo(
                id=plugin_data.get('id', ''),
                name=plugin_data.get('name', ''),
                version=plugin_data.get('version', '1.0.0'),
                description=plugin_data.get('description', ''),
                author=plugin_data.get('author', ''),
                plugin_type=PluginType(plugin_data.get('type', 'component')),
                dependencies=plugin_data.get('dependencies', []),
                entry_point=plugin_data.get('entry_point', ''),
                config_schema=plugin_data.get('config_schema', {}),
                api_endpoints=plugin_data.get('api_endpoints', []),
                permissions=plugin_data.get('permissions', []),
                tags=plugin_data.get('tags', []),
                min_system_version=plugin_data.get('min_system_version', '1.0.0'),
                max_system_version=plugin_data.get('max_system_version', '*'),
                license=plugin_data.get('license', 'MIT'),
                repository=plugin_data.get('repository', ''),
                documentation=plugin_data.get('documentation', '')
            )
            
            return plugin_info
        
        except Exception as e:
            logger.error(f"Error loading manifest {manifest_path}: {e}")
            return None
    
    async def _auto_discover_plugin(self, plugin_path: Path) -> Optional[PluginInfo]:
        """Auto-discover plugin information from directory structure."""
        try:
            # Look for __init__.py or main.py
            init_file = plugin_path / "__init__.py"
            main_file = plugin_path / "main.py"
            
            if not init_file.exists() and not main_file.exists():
                return None
            
            # Try to import the module
            module_name = plugin_path.name
            sys.path.insert(0, str(plugin_path.parent))
            
            try:
                module = importlib.import_module(module_name)
                
                # Look for plugin metadata
                plugin_info = PluginInfo(
                    id=module_name,
                    name=getattr(module, '__name__', module_name),
                    version=getattr(module, '__version__', '1.0.0'),
                    description=getattr(module, '__doc__', ''),
                    author=getattr(module, '__author__', 'Unknown'),
                    plugin_type=PluginType.COMPONENT,
                    entry_point=f"{module_name}.main"
                )
                
                return plugin_info
            
            finally:
                sys.path.remove(str(plugin_path.parent))
        
        except Exception as e:
            logger.error(f"Error auto-discovering plugin {plugin_path}: {e}")
            return None
    
    def _validate_manifest(self, manifest_data: Dict[str, Any]) -> bool:
        """Validate plugin manifest schema."""
        required_fields = ['plugin']
        plugin_required = ['id', 'name', 'version', 'author']
        
        if not all(field in manifest_data for field in required_fields):
            return False
        
        plugin_data = manifest_data.get('plugin', {})
        if not all(field in plugin_data for field in plugin_required):
            return False
        
        return True

class PluginLoader:
    """Plugin loading and instantiation system."""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.loaded_plugins: Dict[str, PluginInstance] = {}
        self._lock = threading.RLock()
    
    async def load_plugin(
        self,
        plugin_info: PluginInfo,
        config: Dict[str, Any] = None
    ) -> PluginInstance:
        """Load a plugin and create its components."""
        
        with self._lock:
            if plugin_info.id in self.loaded_plugins:
                return self.loaded_plugins[plugin_info.id]
            
            plugin_instance = PluginInstance(
                plugin_info=plugin_info,
                status=PluginStatus.LOADED,
                config=config or {}
            )
            
            try:
                # Load the plugin module
                module = await self._load_plugin_module(plugin_info)
                plugin_instance.module = module
                
                # Create components
                components = await self._create_plugin_components(plugin_info, module, config)
                plugin_instance.components = components
                
                # Register components
                for component in components:
                    metadata = component.get_metadata()
                    component_config = ComponentConfig(
                        enabled=True,
                        priority=ComponentPriority.NORMAL,
                        config=config or {}
                    )
                    
                    self.registry.register_component(component, metadata, component_config)
                
                plugin_instance.loaded_at = datetime.now()
                self.loaded_plugins[plugin_info.id] = plugin_instance
                
                logger.info(f"Loaded plugin: {plugin_info.name}")
                return plugin_instance
            
            except Exception as e:
                plugin_instance.status = PluginStatus.ERROR
                plugin_instance.error_message = str(e)
                logger.error(f"Failed to load plugin {plugin_info.name}: {e}")
                return plugin_instance
    
    async def _load_plugin_module(self, plugin_info: PluginInfo) -> Any:
        """Load the plugin module."""
        try:
            if plugin_info.entry_point:
                # Load from entry point
                module_path, class_name = plugin_info.entry_point.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            else:
                # Load from plugin ID
                module = importlib.import_module(plugin_info.id)
                return module
        
        except Exception as e:
            raise ImportError(f"Failed to import plugin module: {e}")
    
    async def _create_plugin_components(
        self,
        plugin_info: PluginInfo,
        module: Any,
        config: Dict[str, Any]
    ) -> List[BaseComponent]:
        """Create plugin components."""
        components = []
        
        try:
            # Look for component classes in the module
            if hasattr(module, 'create_components'):
                # Plugin provides a factory function
                component_classes = module.create_components()
            else:
                # Auto-discover component classes
                component_classes = self._discover_component_classes(module)
            
            # Instantiate components
            for component_class in component_classes:
                if issubclass(component_class, BaseComponent):
                    component = component_class()
                    components.append(component)
            
            return components
        
        except Exception as e:
            logger.error(f"Error creating plugin components: {e}")
            return []
    
    def _discover_component_classes(self, module: Any) -> List[Type[BaseComponent]]:
        """Discover component classes in a module."""
        component_classes = []
        
        for name in dir(module):
            obj = getattr(module, name)
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseComponent) and 
                obj != BaseComponent):
                component_classes.append(obj)
        
        return component_classes
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin and its components."""
        with self._lock:
            if plugin_id not in self.loaded_plugins:
                return False
            
            plugin_instance = self.loaded_plugins[plugin_id]
            
            try:
                # Stop and unregister components
                for component in plugin_instance.components:
                    await component.stop()
                    self.registry.unregister_component(component.id)
                
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_id]
                
                logger.info(f"Unloaded plugin: {plugin_instance.plugin_info.name}")
                return True
            
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_id}: {e}")
                return False

class PluginManager:
    """Main plugin management system."""
    
    def __init__(
        self,
        plugin_directories: List[str] = None,
        registry: ComponentRegistry = None,
        event_bus = None
    ):
        self.plugin_directories = plugin_directories or [
            "plugins",
            "extensions",
            "components"
        ]
        
        self.registry = registry or get_global_registry()
        self.event_bus = event_bus or get_global_event_bus()
        
        self.discovery = PluginDiscovery(self.plugin_directories)
        self.loader = PluginLoader(self.registry)
        
        self.enabled_plugins: Dict[str, PluginInstance] = {}
        self._lock = threading.RLock()
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        logger.info("Initializing Plugin Manager")
        
        # Discover available plugins
        await self.discovery.discover_plugins()
        
        # Load enabled plugins
        await self._load_enabled_plugins()
        
        logger.info("Plugin Manager initialized")
    
    async def discover_plugins(self) -> Dict[str, PluginInfo]:
        """Discover all available plugins."""
        return await self.discovery.discover_plugins()
    
    async def load_plugin(
        self,
        plugin_id: str,
        config: Dict[str, Any] = None
    ) -> bool:
        """Load a plugin."""
        plugin_info = self.discovery.discovered_plugins.get(plugin_id)
        if not plugin_info:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        try:
            plugin_instance = await self.loader.load_plugin(plugin_info, config)
            
            if plugin_instance.status == PluginStatus.LOADED:
                # Initialize and start components
                for component in plugin_instance.components:
                    await component.initialize(config)
                    await component.start()
                
                plugin_instance.status = PluginStatus.ENABLED
                plugin_instance.enabled_at = datetime.now()
                
                with self._lock:
                    self.enabled_plugins[plugin_id] = plugin_instance
                
                # Publish event
                await self.event_bus.publish(ComponentEvent(
                    event_type="plugin.enabled",
                    component_id=plugin_id,
                    data={"name": plugin_info.name, "version": plugin_info.version}
                ))
                
                logger.info(f"Enabled plugin: {plugin_info.name}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        with self._lock:
            if plugin_id not in self.enabled_plugins:
                return False
            
            plugin_instance = self.enabled_plugins[plugin_id]
        
        try:
            # Stop components
            for component in plugin_instance.components:
                await component.stop()
            
            # Unload plugin
            success = await self.loader.unload_plugin(plugin_id)
            
            if success:
                with self._lock:
                    del self.enabled_plugins[plugin_id]
                
                # Publish event
                await self.event_bus.publish(ComponentEvent(
                    event_type="plugin.disabled",
                    component_id=plugin_id,
                    data={"name": plugin_instance.plugin_info.name}
                ))
                
                logger.info(f"Disabled plugin: {plugin_instance.plugin_info.name}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    async def reload_plugin(self, plugin_id: str) -> bool:
        """Reload a plugin."""
        # Unload first
        await self.unload_plugin(plugin_id)
        
        # Reload
        return await self.load_plugin(plugin_id)
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInstance]:
        """Get a plugin instance."""
        return self.enabled_plugins.get(plugin_id)
    
    def list_plugins(self, status: Optional[PluginStatus] = None) -> List[PluginInstance]:
        """List plugins with optional status filter."""
        plugins = list(self.enabled_plugins.values())
        
        if status:
            plugins = [p for p in plugins if p.status == status]
        
        return plugins
    
    def get_plugin_health(self) -> Dict[str, Any]:
        """Get plugin system health status."""
        total_plugins = len(self.enabled_plugins)
        healthy_plugins = len([p for p in self.enabled_plugins.values() 
                              if p.status == PluginStatus.ENABLED])
        
        return {
            "total_plugins": total_plugins,
            "healthy_plugins": healthy_plugins,
            "error_plugins": total_plugins - healthy_plugins,
            "plugin_directories": self.plugin_directories,
            "discovered_plugins": len(self.discovery.discovered_plugins)
        }
    
    async def _load_enabled_plugins(self) -> None:
        """Load plugins that are marked as enabled."""
        # This would typically read from a configuration file
        # For now, we'll load all discovered plugins
        for plugin_id, plugin_info in self.discovery.discovered_plugins.items():
            try:
                await self.load_plugin(plugin_id)
            except Exception as e:
                logger.error(f"Failed to auto-load plugin {plugin_id}: {e}")

# Global plugin manager instance
_global_plugin_manager: Optional[PluginManager] = None

def get_global_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager



























