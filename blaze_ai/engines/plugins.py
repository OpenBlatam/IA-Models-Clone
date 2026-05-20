"""
Plugin System for Blaze AI Engines.

This module provides a comprehensive plugin system for dynamically
loading and managing engine plugins, extensions, and custom implementations.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union
from zipfile import ZipFile
import tempfile
import shutil

from .base import Engine, EngineType, EnginePriority
from .factory import EngineTemplate
from ..utils.logging import get_logger

# =============================================================================
# Plugin System Configuration
# =============================================================================

@dataclass
class PluginConfig:
    """Configuration for the plugin system."""
    plugin_directories: List[str] = None
    enable_hot_reload: bool = False
    hot_reload_interval: float = 30.0
    enable_plugin_validation: bool = True
    allow_unsafe_plugins: bool = False
    plugin_cache_size: int = 100
    enable_plugin_metrics: bool = True
    plugin_timeout: float = 30.0
    
    def __post_init__(self):
        if self.plugin_directories is None:
            self.plugin_directories = ["plugins", "extensions", "custom_engines"]

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    tags: List[str] = None
    dependencies: List[str] = None
    requirements: Dict[str, Any] = None
    engine_types: List[str] = None
    priority: EnginePriority = EnginePriority.NORMAL
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.requirements is None:
            self.requirements = {}
        if self.engine_types is None:
            self.engine_types = []
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()
        self.updated_at = self.created_at

@dataclass
class PluginInfo:
    """Complete information about a plugin."""
    metadata: PluginMetadata
    plugin_path: Path
    is_loaded: bool = False
    load_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    engine_templates: List[EngineTemplate] = field(default_factory=list)

# =============================================================================
# Plugin Loader Implementation
# =============================================================================

class PluginLoader:
    """Plugin loader for dynamic engine loading."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self.logger = get_logger("plugin_loader")
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_cache: Dict[str, Any] = {}
        self._plugin_watchers: List[Callable] = []
        
        # Initialize plugin system
        self._initialize_plugin_system()
    
    def _initialize_plugin_system(self):
        """Initialize the plugin system."""
        self.logger.info("Initializing plugin system...")
        
        # Create plugin directories if they don't exist
        for directory in self.config.plugin_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Scan for existing plugins
        self._scan_plugins()
        
        self.logger.info(f"Plugin system initialized with {len(self.plugins)} plugins")
    
    def _scan_plugins(self):
        """Scan for available plugins in configured directories."""
        for directory in self.config.plugin_directories:
            plugin_dir = Path(directory)
            if plugin_dir.exists():
                self._scan_directory_for_plugins(plugin_dir)
    
    def _scan_directory_for_plugins(self, directory: Path):
        """Scan a specific directory for plugins."""
        for item in directory.iterdir():
            if item.is_file():
                if item.suffix in ['.py', '.zip']:
                    self._try_load_plugin(item)
            elif item.is_dir():
                # Check for Python packages
                if (item / '__init__.py').exists():
                    self._try_load_plugin(item)
                # Check for plugin directories
                elif (item / 'plugin.json').exists():
                    self._try_load_plugin(item)
    
    def _try_load_plugin(self, plugin_path: Path):
        """Try to load a plugin from a path."""
        try:
            plugin_name = plugin_path.stem
            if plugin_name in self.plugins:
                self.logger.debug(f"Plugin {plugin_name} already loaded, skipping")
                return
            
            # Load plugin metadata
            metadata = self._extract_plugin_metadata(plugin_path)
            if not metadata:
                return
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                plugin_path=plugin_path
            )
            
            # Try to load the plugin
            if self._load_plugin_engines(plugin_info):
                self.plugins[plugin_name] = plugin_info
                self.logger.info(f"Successfully loaded plugin: {plugin_name}")
            else:
                self.logger.warning(f"Failed to load plugin: {plugin_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_path}: {e}")
    
    def _extract_plugin_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Extract metadata from a plugin."""
        try:
            # Try to load plugin.json first
            if plugin_path.is_dir() and (plugin_path / 'plugin.json').exists():
                with open(plugin_path / 'plugin.json', 'r') as f:
                    metadata_dict = json.load(f)
                    return PluginMetadata(**metadata_dict)
            
            # Try to extract from Python file/docstring
            elif plugin_path.suffix == '.py':
                return self._extract_metadata_from_python(plugin_path)
            
            # Try to extract from ZIP file
            elif plugin_path.suffix == '.zip':
                return self._extract_metadata_from_zip(plugin_path)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {plugin_path}: {e}")
        
        return None
    
    def _extract_metadata_from_python(self, python_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from a Python file."""
        try:
            # Read the file and look for metadata
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for metadata in docstring or comments
            metadata = {
                'name': python_file.stem,
                'version': '1.0.0',
                'description': f'Plugin loaded from {python_file.name}',
                'author': 'Unknown',
                'license': 'Unknown',
                'homepage': '',
                'repository': '',
                'tags': [],
                'dependencies': [],
                'requirements': {},
                'engine_types': [],
                'priority': EnginePriority.NORMAL
            }
            
            # Try to find more metadata in the file
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                    # Parse metadata from comments/docstrings
                    if 'version:' in line.lower():
                        metadata['version'] = line.split(':')[1].strip()
                    elif 'author:' in line.lower():
                        metadata['author'] = line.split(':')[1].strip()
                    elif 'description:' in line.lower():
                        metadata['description'] = line.split(':')[1].strip()
            
            return PluginMetadata(**metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from Python file {python_file}: {e}")
            return None
    
    def _extract_metadata_from_zip(self, zip_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from a ZIP file."""
        try:
            with ZipFile(zip_file, 'r') as zip_ref:
                # Look for plugin.json in the ZIP
                if 'plugin.json' in zip_ref.namelist():
                    with zip_ref.open('plugin.json') as f:
                        metadata_dict = json.load(f)
                        return PluginMetadata(**metadata_dict)
                
                # Look for Python files
                python_files = [f for f in zip_ref.namelist() if f.endswith('.py')]
                if python_files:
                    # Use the first Python file to extract metadata
                    with zip_ref.open(python_files[0]) as f:
                        content = f.read().decode('utf-8')
                        # Create basic metadata
                        metadata = {
                            'name': zip_file.stem,
                            'version': '1.0.0',
                            'description': f'Plugin loaded from {zip_file.name}',
                            'author': 'Unknown',
                            'license': 'Unknown',
                            'homepage': '',
                            'repository': '',
                            'tags': [],
                            'dependencies': [],
                            'requirements': {},
                            'engine_types': [],
                            'priority': EnginePriority.NORMAL
                        }
                        return PluginMetadata(**metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from ZIP file {zip_file}: {e}")
        
        return None
    
    def _load_plugin_engines(self, plugin_info: PluginInfo) -> bool:
        """Load engine templates from a plugin."""
        try:
            plugin_path = plugin_info.plugin_path
            
            if plugin_path.is_file() and plugin_path.suffix == '.py':
                # Load from Python file
                return self._load_engines_from_python_file(plugin_path, plugin_info)
            
            elif plugin_path.is_file() and plugin_path.suffix == '.zip':
                # Load from ZIP file
                return self._load_engines_from_zip_file(plugin_path, plugin_info)
            
            elif plugin_path.is_dir():
                # Load from directory
                return self._load_engines_from_directory(plugin_path, plugin_info)
            
        except Exception as e:
            self.logger.error(f"Failed to load engines from plugin {plugin_info.metadata.name}: {e}")
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
        
        return False
    
    def _load_engines_from_python_file(self, python_file: Path, plugin_info: PluginInfo) -> bool:
        """Load engines from a Python file."""
        try:
            # Create a temporary module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_info.metadata.name}", 
                python_file
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for engine classes
                engines_found = self._extract_engines_from_module(module, plugin_info)
                return engines_found > 0
                
        except Exception as e:
            self.logger.error(f"Failed to load Python file {python_file}: {e}")
        
        return False
    
    def _load_engines_from_zip_file(self, zip_file: Path, plugin_info: PluginInfo) -> bool:
        """Load engines from a ZIP file."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Look for Python files in the extracted content
                temp_path = Path(temp_dir)
                python_files = list(temp_path.rglob('*.py'))
                
                engines_found = 0
                for python_file in python_files:
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"plugin_{plugin_info.metadata.name}_{python_file.stem}",
                            python_file
                        )
                        
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            engines_found += self._extract_engines_from_module(module, plugin_info)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to load {python_file} from ZIP: {e}")
                
                return engines_found > 0
                
        except Exception as e:
            self.logger.error(f"Failed to load ZIP file {zip_file}: {e}")
        
        return False
    
    def _load_engines_from_directory(self, directory: Path, plugin_info: PluginInfo) -> bool:
        """Load engines from a directory."""
        try:
            engines_found = 0
            
            # Look for Python files
            python_files = list(directory.rglob('*.py'))
            for python_file in python_files:
                if python_file.name != '__init__.py':
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"plugin_{plugin_info.metadata.name}_{python_file.stem}",
                            python_file
                        )
                        
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            engines_found += self._extract_engines_from_module(module, plugin_info)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to load {python_file}: {e}")
            
            return engines_found > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load directory {directory}: {e}")
        
        return False
    
    def _extract_engines_from_module(self, module, plugin_info: PluginInfo) -> int:
        """Extract engine classes from a module."""
        engines_found = 0
        
        try:
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Engine) and 
                    obj != Engine):
                    
                    # Create engine template
                    template = self._create_engine_template_from_class(obj, plugin_info)
                    if template:
                        plugin_info.engine_templates.append(template)
                        engines_found += 1
                        self.logger.info(f"Found engine {name} in plugin {plugin_info.metadata.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to extract engines from module: {e}")
        
        return engines_found
    
    def _create_engine_template_from_class(self, engine_class: Type[Engine], plugin_info: PluginInfo) -> Optional[EngineTemplate]:
        """Create an engine template from an engine class."""
        try:
            # Get default config if available
            default_config = {}
            if hasattr(engine_class, 'get_default_config'):
                default_config = engine_class.get_default_config()
            
            # Create template
            template = EngineTemplate(
                name=f"{plugin_info.metadata.name}_{engine_class.__name__.lower()}",
                engine_class=engine_class,
                default_config=default_config,
                description=f"{engine_class.__name__} from plugin {plugin_info.metadata.name}",
                tags=plugin_info.metadata.tags + [plugin_info.metadata.name, "plugin"],
                priority=plugin_info.metadata.priority,
                dependencies=plugin_info.metadata.dependencies,
                requirements=plugin_info.metadata.requirements
            )
            
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to create template for {engine_class.__name__}: {e}")
            return None
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        return list(self.plugins.keys())
    
    def get_plugin_engines(self, plugin_name: str) -> List[EngineTemplate]:
        """Get engine templates from a specific plugin."""
        plugin_info = self.plugins.get(plugin_name)
        if plugin_info:
            return plugin_info.engine_templates
        return []
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        plugin_path = plugin_info.plugin_path
        
        # Remove old plugin
        del self.plugins[plugin_name]
        
        # Try to reload
        return self._try_load_plugin(plugin_path)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        if plugin_name not in self.plugins:
            return False
        
        del self.plugins[plugin_name]
        self.logger.info(f"Unloaded plugin: {plugin_name}")
        return True
    
    def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get metrics about all plugins."""
        total_plugins = len(self.plugins)
        loaded_plugins = len([p for p in self.plugins.values() if p.is_loaded])
        total_engines = sum(len(p.engine_templates) for p in self.plugins.values())
        
        return {
            "total_plugins": total_plugins,
            "loaded_plugins": loaded_plugins,
            "failed_plugins": total_plugins - loaded_plugins,
            "total_engines": total_engines,
            "plugins": {
                name: {
                    "is_loaded": info.is_loaded,
                    "engine_count": len(info.engine_templates),
                    "error_count": info.error_count,
                    "last_error": info.last_error
                }
                for name, info in self.plugins.items()
            }
        }

# =============================================================================
# Plugin Manager
# =============================================================================

class PluginManager:
    """High-level plugin manager for the Blaze AI system."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self.logger = get_logger("plugin_manager")
        self.loader = PluginLoader(config)
        self._watchers: List[Callable] = []
    
    def register_plugin_watcher(self, callback: Callable):
        """Register a callback to be notified of plugin changes."""
        self._watchers.append(callback)
    
    def unregister_plugin_watcher(self, callback: Callable):
        """Unregister a plugin watcher callback."""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def _notify_watchers(self, event: str, plugin_name: str, **kwargs):
        """Notify all watchers of a plugin event."""
        for watcher in self._watchers:
            try:
                watcher(event, plugin_name, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in plugin watcher {watcher}: {e}")
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a plugin from a path."""
        try:
            plugin_file = Path(plugin_path)
            if not plugin_file.exists():
                self.logger.error(f"Plugin file not found: {plugin_path}")
                return False
            
            # Copy to plugins directory
            target_dir = Path(self.config.plugin_directories[0])
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if plugin_file.is_file():
                target_path = target_dir / plugin_file.name
                shutil.copy2(plugin_file, target_path)
            else:
                target_path = target_dir / plugin_file.name
                shutil.copytree(plugin_file, target_path, dirs_exist_ok=True)
            
            # Try to load the plugin
            if self.loader._try_load_plugin(target_path):
                self._notify_watchers("plugin_installed", target_path.stem)
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to install plugin {plugin_path}: {e}")
        
        return False
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """Remove a plugin completely."""
        try:
            plugin_info = self.loader.get_plugin_info(plugin_name)
            if not plugin_info:
                return False
            
            # Unload from memory
            self.loader.unload_plugin(plugin_name)
            
            # Remove from filesystem
            if plugin_info.plugin_path.exists():
                if plugin_info.plugin_path.is_file():
                    plugin_info.plugin_path.unlink()
                else:
                    shutil.rmtree(plugin_info.plugin_path)
            
            self._notify_watchers("plugin_removed", plugin_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove plugin {plugin_name}: {e}")
        
        return False
    
    def get_all_engine_templates(self) -> List[EngineTemplate]:
        """Get all engine templates from all plugins."""
        templates = []
        for plugin_name in self.loader.list_plugins():
            templates.extend(self.loader.get_plugin_engines(plugin_name))
        return templates
    
    def search_plugins(self, query: str) -> List[str]:
        """Search plugins by name, description, or tags."""
        results = []
        query_lower = query.lower()
        
        for plugin_name, plugin_info in self.loader.plugins.items():
            metadata = plugin_info.metadata
            
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(plugin_name)
        
        return results

# =============================================================================
# Factory Functions
# =============================================================================

def create_plugin_manager(config: Optional[PluginConfig] = None) -> PluginManager:
    """Create a plugin manager with custom configuration."""
    return PluginManager(config)

def create_standard_plugin_manager() -> PluginManager:
    """Create a standard plugin manager with default configuration."""
    config = PluginConfig(
        enable_hot_reload=True,
        enable_plugin_validation=True,
        allow_unsafe_plugins=False
    )
    return PluginManager(config)

# =============================================================================
# Global Plugin Manager Instance
# =============================================================================

_default_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _default_plugin_manager
    if _default_plugin_manager is None:
        _default_plugin_manager = create_standard_plugin_manager()
    return _default_plugin_manager

