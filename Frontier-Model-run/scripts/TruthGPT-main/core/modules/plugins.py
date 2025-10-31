"""
Plugin System
Extensible plugin architecture for modular components
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional, Type, Callable, Protocol
from pathlib import Path
import json
import time
from dataclasses import dataclass, field

from .interfaces import IPlugin, IConfigurable, ILoggable, IMeasurable
from .base import BaseModule, ModuleRegistry

logger = logging.getLogger(__name__)

@dataclass
class PluginInfo:
    """Plugin information"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    path: str = ""
    loaded_at: float = field(default_factory=time.time)
    enabled: bool = True

class PluginLoader:
    """Plugin loader for dynamic loading"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: Dict[str, Type[BaseModule]] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        
    def add_plugin_dir(self, directory: str) -> None:
        """Add plugin directory"""
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            sys.path.insert(0, directory)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins"""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            for file_path in Path(plugin_dir).glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                    
                module_name = file_path.stem
                try:
                    # Try to load the module
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Check if module has plugin class
                        plugin_class = self._find_plugin_class(module)
                        if plugin_class:
                            discovered.append(module_name)
                            
                except Exception as e:
                    logger.warning(f"Failed to discover plugin {file_path}: {e}")
        
        return discovered
    
    def _find_plugin_class(self, module) -> Optional[Type[BaseModule]]:
        """Find plugin class in module"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseModule) and 
                obj != BaseModule and
                hasattr(obj, 'get_name')):
                return obj
        return None
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseModule]:
        """Load a specific plugin"""
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = os.path.join(plugin_dir, f"{plugin_name}.py")
            if os.path.exists(plugin_path):
                try:
                    # Load the module
                    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find plugin class
                        plugin_class = self._find_plugin_class(module)
                        if plugin_class:
                            # Create plugin info
                            info = PluginInfo(
                                name=plugin_name,
                                version=getattr(plugin_class, 'version', '1.0.0'),
                                description=getattr(plugin_class, 'description', ''),
                                author=getattr(plugin_class, 'author', ''),
                                dependencies=getattr(plugin_class, 'dependencies', []),
                                tags=getattr(plugin_class, 'tags', []),
                                path=plugin_path
                            )
                            
                            # Create plugin instance
                            instance = plugin_class(plugin_name, config or {})
                            self.loaded_plugins[plugin_name] = instance
                            self.plugin_info[plugin_name] = info
                            
                            logger.info(f"Loaded plugin: {plugin_name}")
                            return instance
                            
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")
                    return None
        
        logger.error(f"Plugin not found: {plugin_name}")
        return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            del self.loaded_plugins[plugin_name]
            del self.plugin_info[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.plugin_info.get(plugin_name)
    
    def list_loaded_plugins(self) -> List[str]:
        """List loaded plugins"""
        return list(self.loaded_plugins.keys())

class PluginManager:
    """Manager for plugin system"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.loader = PluginLoader(plugin_dirs)
        self.registry = ModuleRegistry()
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
    def add_plugin_directory(self, directory: str) -> None:
        """Add plugin directory"""
        self.loader.add_plugin_dir(directory)
    
    def discover_and_load_plugins(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, bool]:
        """Discover and load all available plugins"""
        discovered = self.loader.discover_plugins()
        results = {}
        
        for plugin_name in discovered:
            plugin_config = config.get(plugin_name, {}) if config else {}
            plugin = self.loader.load_plugin(plugin_name, plugin_config)
            
            if plugin:
                # Register in module registry
                self.registry.register(plugin)
                results[plugin_name] = True
            else:
                results[plugin_name] = False
        
        return results
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a specific plugin"""
        plugin = self.loader.load_plugin(plugin_name, config)
        if plugin:
            self.registry.register(plugin)
            self.plugin_configs[plugin_name] = config or {}
            return True
        return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if self.registry.unregister(plugin_name):
            return self.loader.unload_plugin(plugin_name)
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BaseModule]:
        """Get a loaded plugin"""
        return self.registry.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all plugins"""
        return self.registry.list_modules()
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.loader.get_plugin_info(plugin_name)
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin"""
        plugin = self.get_plugin(plugin_name)
        if plugin and hasattr(plugin, 'configure'):
            return plugin.configure(config)
        return False
    
    def start_plugin(self, plugin_name: str) -> bool:
        """Start a plugin"""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.start()
        return False
    
    def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a plugin"""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.stop()
        return False
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins"""
        status = {}
        for plugin_name in self.list_plugins():
            plugin = self.get_plugin(plugin_name)
            if plugin:
                status[plugin_name] = {
                    "state": plugin.get_state().value,
                    "healthy": plugin.is_healthy(),
                    "info": self.get_plugin_info(plugin_name).__dict__ if self.get_plugin_info(plugin_name) else {}
                }
        return status

class PluginRegistry:
    """Registry for plugin metadata and configurations"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInfo] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, info: PluginInfo) -> None:
        """Register plugin information"""
        self.plugins[info.name] = info
    
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.plugins.get(name)
    
    def set_plugin_config(self, name: str, config: Dict[str, Any]) -> None:
        """Set plugin configuration"""
        self.configurations[name] = config
    
    def get_plugin_config(self, name: str) -> Dict[str, Any]:
        """Get plugin configuration"""
        return self.configurations.get(name, {})
    
    def list_plugins(self) -> List[str]:
        """List registered plugins"""
        return list(self.plugins.keys())
    
    def get_plugins_by_tag(self, tag: str) -> List[str]:
        """Get plugins by tag"""
        return [name for name, info in self.plugins.items() if tag in info.tags]
    
    def get_plugins_by_dependency(self, dependency: str) -> List[str]:
        """Get plugins that depend on a specific dependency"""
        return [name for name, info in self.plugins.items() if dependency in info.dependencies]

class PluginValidator:
    """Validator for plugin compatibility and requirements"""
    
    def __init__(self):
        self.requirements: Dict[str, List[str]] = {}
        self.compatibility_matrix: Dict[str, Dict[str, bool]] = {}
    
    def add_requirement(self, plugin_name: str, requirements: List[str]) -> None:
        """Add requirements for a plugin"""
        self.requirements[plugin_name] = requirements
    
    def check_requirements(self, plugin_name: str, available_modules: List[str]) -> bool:
        """Check if plugin requirements are met"""
        if plugin_name not in self.requirements:
            return True
        
        required = self.requirements[plugin_name]
        return all(req in available_modules for req in required)
    
    def set_compatibility(self, plugin1: str, plugin2: str, compatible: bool) -> None:
        """Set compatibility between plugins"""
        if plugin1 not in self.compatibility_matrix:
            self.compatibility_matrix[plugin1] = {}
        self.compatibility_matrix[plugin1][plugin2] = compatible
    
    def are_compatible(self, plugin1: str, plugin2: str) -> bool:
        """Check if two plugins are compatible"""
        if plugin1 not in self.compatibility_matrix:
            return True
        return self.compatibility_matrix[plugin1].get(plugin2, True)
    
    def validate_plugin_set(self, plugins: List[str]) -> Dict[str, List[str]]:
        """Validate a set of plugins for compatibility"""
        issues = {}
        
        for i, plugin1 in enumerate(plugins):
            plugin_issues = []
            
            # Check requirements
            if not self.check_requirements(plugin1, plugins):
                missing = [req for req in self.requirements.get(plugin1, []) if req not in plugins]
                plugin_issues.append(f"Missing requirements: {missing}")
            
            # Check compatibility
            for plugin2 in plugins[i+1:]:
                if not self.are_compatible(plugin1, plugin2):
                    plugin_issues.append(f"Incompatible with {plugin2}")
            
            if plugin_issues:
                issues[plugin1] = plugin_issues
        
        return issues

# Example plugin base class
class BasePlugin(BaseModule):
    """Base class for plugins"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.version = "1.0.0"
        self.description = "Base plugin"
        self.author = "Unknown"
        self.dependencies = []
        self.tags = []
    
    def get_name(self) -> str:
        """Get plugin name"""
        return self.name
    
    def get_version(self) -> str:
        """Get plugin version"""
        return self.version
    
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies"""
        return self.dependencies
    
    def is_compatible(self, version: str) -> bool:
        """Check compatibility"""
        return True
    
    def initialize(self) -> bool:
        """Initialize plugin"""
        self.set_state(self.ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start plugin"""
        self.set_state(self.ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop plugin"""
        self.set_state(self.ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup plugin"""
        return True

