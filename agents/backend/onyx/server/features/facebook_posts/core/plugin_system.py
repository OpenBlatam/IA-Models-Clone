#!/usr/bin/env python3
"""
Plugin System - Ultra-Modular Architecture v3.7
Dynamic plugin loading and management system
"""
import os
import sys
import json
import importlib
import inspect
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Union
from datetime import datetime
import threading
import logging
import shutil

from .base_system import BaseModule, ModuleConfig, ModulePriority, create_module_config

logger = logging.getLogger(__name__)

class PluginInfo:
    """Information about a plugin"""
    
    def __init__(self, name: str, version: str, description: str = "", author: str = "", 
                 dependencies: List[str] = None, config_schema: Dict = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}
        self.loaded_at = datetime.now()
        self.enabled = True
        self.error_count = 0
        self.last_error = None

class PluginManager:
    """
    Manages dynamic plugin loading and lifecycle
    Supports hot-reloading, dependency management, and plugin discovery
    """
    
    def __init__(self, plugins_path: str = "plugins", temp_path: str = "temp_plugins"):
        """Initialize plugin manager"""
        self.plugins_path = Path(plugins_path)
        self.temp_path = Path(temp_path)
        self.plugins: Dict[str, BaseModule] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.plugin_classes: Dict[str, Type[BaseModule]] = {}
        
        # Plugin lifecycle
        self._loaded_plugins: List[str] = []
        self._enabled_plugins: List[str] = []
        self._disabled_plugins: List[str] = []
        self._error_plugins: List[str] = []
        
        # Plugin discovery
        self._plugin_metadata: Dict[str, Dict] = {}
        self._plugin_dependencies: Dict[str, List[str]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._plugin_watcher = None
        self._watch_enabled = False
        
        # Create directories
        self.plugins_path.mkdir(exist_ok=True)
        self.temp_path.mkdir(exist_ok=True)
        
        # Initialize
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Discover available plugins"""
        try:
            logger.info(f"Discovering plugins in: {self.plugins_path}")
            
            # Scan for plugin directories and files
            for item in self.plugins_path.iterdir():
                if item.is_dir():
                    self._discover_plugin_directory(item)
                elif item.is_file() and item.suffix in ['.py', '.zip']:
                    self._discover_plugin_file(item)
            
            logger.info(f"Discovered {len(self._plugin_metadata)} plugins")
            
        except Exception as e:
            logger.error(f"Error during plugin discovery: {e}")
    
    def _discover_plugin_directory(self, plugin_dir: Path):
        """Discover plugin in a directory"""
        try:
            # Look for plugin manifest
            manifest_file = plugin_dir / "plugin.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                plugin_name = manifest.get('name', plugin_dir.name)
                self._plugin_metadata[plugin_name] = {
                    'type': 'directory',
                    'path': str(plugin_dir),
                    'manifest': manifest
                }
                
                # Look for main plugin file
                main_file = plugin_dir / f"{plugin_name}.py"
                if main_file.exists():
                    self._plugin_classes[plugin_name] = self._load_plugin_class(main_file, plugin_name)
            
            # Look for Python files directly
            else:
                for py_file in plugin_dir.glob("*.py"):
                    if py_file.name.startswith("__"):
                        continue
                    
                    plugin_name = py_file.stem
                    if plugin_name not in self._plugin_metadata:
                        self._plugin_metadata[plugin_name] = {
                            'type': 'file',
                            'path': str(py_file)
                        }
                        self._plugin_classes[plugin_name] = self._load_plugin_class(py_file, plugin_name)
                        
        except Exception as e:
            logger.warning(f"Error discovering plugin directory {plugin_dir}: {e}")
    
    def _discover_plugin_file(self, plugin_file: Path):
        """Discover plugin from a file"""
        try:
            if plugin_file.suffix == '.py':
                plugin_name = plugin_file.stem
                self._plugin_metadata[plugin_name] = {
                    'type': 'file',
                    'path': str(plugin_file)
                }
                self._plugin_classes[plugin_name] = self._load_plugin_class(plugin_file, plugin_name)
            
            elif plugin_file.suffix == '.zip':
                self._discover_zip_plugin(plugin_file)
                
        except Exception as e:
            logger.warning(f"Error discovering plugin file {plugin_file}: {e}")
    
    def _discover_zip_plugin(self, zip_file: Path):
        """Discover plugin from a ZIP file"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Look for plugin.json in ZIP
                if 'plugin.json' in zip_ref.namelist():
                    with zip_ref.open('plugin.json') as f:
                        manifest = json.load(f)
                    
                    plugin_name = manifest.get('name', zip_file.stem)
                    self._plugin_metadata[plugin_name] = {
                        'type': 'zip',
                        'path': str(zip_file),
                        'manifest': manifest
                    }
                    
                    # Extract to temp directory for loading
                    temp_dir = self.temp_path / plugin_name
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    
                    zip_ref.extractall(temp_dir)
                    
                    # Look for main plugin file
                    main_file = temp_dir / f"{plugin_name}.py"
                    if main_file.exists():
                        self._plugin_classes[plugin_name] = self._load_plugin_class(main_file, plugin_name)
                        
        except Exception as e:
            logger.warning(f"Error discovering ZIP plugin {zip_file}: {e}")
    
    def _load_plugin_class(self, plugin_file: Path, plugin_name: str) -> Optional[Type[BaseModule]]:
        """Load plugin class from file"""
        try:
            # Add plugin directory to path
            plugin_dir = str(plugin_file.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import module
            module = importlib.import_module(plugin_name)
            
            # Look for BaseModule subclasses
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseModule) and 
                    obj != BaseModule):
                    
                    logger.info(f"Loaded plugin class: {name} from {plugin_name}")
                    return obj
            
            return None
            
        except Exception as e:
            logger.warning(f"Error loading plugin class from {plugin_file}: {e}")
            return None
    
    def load_plugin(self, plugin_name: str, config: Optional[ModuleConfig] = None) -> Optional[BaseModule]:
        """Load a specific plugin"""
        try:
            if plugin_name not in self._plugin_classes:
                logger.error(f"Plugin class not found: {plugin_name}")
                return None
            
            if plugin_name in self.plugins:
                logger.warning(f"Plugin already loaded: {plugin_name}")
                return self.plugins[plugin_name]
            
            # Get plugin metadata
            metadata = self._plugin_metadata.get(plugin_name, {})
            manifest = metadata.get('manifest', {})
            
            # Create plugin info
            plugin_info = PluginInfo(
                name=plugin_name,
                version=manifest.get('version', '1.0.0'),
                description=manifest.get('description', f'Plugin {plugin_name}'),
                author=manifest.get('author', 'Unknown'),
                dependencies=manifest.get('dependencies', []),
                config_schema=manifest.get('config_schema', {})
            )
            
            # Create default config if none provided
            if config is None:
                config = create_module_config(
                    name=plugin_name,
                    version=plugin_info.version,
                    description=plugin_info.description,
                    dependencies=plugin_info.dependencies,
                    auto_start=False
                )
            
            # Create plugin instance
            plugin_class = self._plugin_classes[plugin_name]
            plugin = plugin_class(config)
            
            # Store plugin
            self.plugins[plugin_name] = plugin
            self.plugin_info[plugin_name] = plugin_info
            
            # Update plugin lists
            if plugin_name not in self._loaded_plugins:
                self._loaded_plugins.append(plugin_name)
            
            if plugin_info.enabled:
                if plugin_name not in self._enabled_plugins:
                    self._enabled_plugins.append(plugin_name)
            else:
                if plugin_name not in self._disabled_plugins:
                    self._disabled_plugins.append(plugin_name)
            
            logger.info(f"Plugin loaded successfully: {plugin_name}")
            return plugin
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            
            # Update error tracking
            if plugin_name in self.plugin_info:
                self.plugin_info[plugin_name].error_count += 1
                self.plugin_info[plugin_name].last_error = str(e)
            
            if plugin_name not in self._error_plugins:
                self._error_plugins.append(plugin_name)
            
            return None
    
    def load_plugins_from_config(self, config_file: str) -> Dict[str, bool]:
        """Load multiple plugins from configuration file"""
        results = {}
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for plugin_name, plugin_config in config_data.get('plugins', {}).items():
                try:
                    # Create plugin config
                    config = create_module_config(
                        name=plugin_name,
                        version=plugin_config.get('version', '1.0.0'),
                        description=plugin_config.get('description', f'Plugin {plugin_name}'),
                        priority=ModulePriority(plugin_config.get('priority', 50)),
                        enabled=plugin_config.get('enabled', True),
                        auto_start=plugin_config.get('auto_start', False),
                        dependencies=plugin_config.get('dependencies', []),
                        log_level=plugin_config.get('log_level', 'INFO')
                    )
                    
                    # Load plugin
                    plugin = self.load_plugin(plugin_name, config)
                    results[plugin_name] = plugin is not None
                    
                except Exception as e:
                    logger.error(f"Error loading plugin {plugin_name}: {e}")
                    results[plugin_name] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading plugins from config: {e}")
            return {}
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        try:
            if plugin_name not in self.plugin_info:
                logger.error(f"Plugin not found: {plugin_name}")
                return False
            
            plugin_info = self.plugin_info[plugin_name]
            plugin_info.enabled = True
            
            # Update plugin lists
            if plugin_name in self._disabled_plugins:
                self._disabled_plugins.remove(plugin_name)
            if plugin_name not in self._enabled_plugins:
                self._enabled_plugins.append(plugin_name)
            
            logger.info(f"Plugin enabled: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        try:
            if plugin_name not in self.plugin_info:
                logger.error(f"Plugin not found: {plugin_name}")
                return False
            
            plugin_info = self.plugin_info[plugin_name]
            plugin_info.enabled = False
            
            # Update plugin lists
            if plugin_name in self._enabled_plugins:
                self._enabled_plugins.remove(plugin_name)
            if plugin_name not in self._disabled_plugins:
                self._disabled_plugins.append(plugin_name)
            
            # Stop plugin if running
            if plugin_name in self.plugins:
                plugin = self.plugins[plugin_name]
                if plugin.status.state.value == 'running':
                    plugin.stop()
            
            logger.info(f"Plugin disabled: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        try:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin not loaded: {plugin_name}")
                return False
            
            logger.info(f"Reloading plugin: {plugin_name}")
            
            # Stop plugin if running
            plugin = self.plugins[plugin_name]
            if plugin.status.state.value == 'running':
                plugin.stop()
            
            # Unload plugin
            self.unload_plugin(plugin_name)
            
            # Reload plugin
            success = self.load_plugin(plugin_name) is not None
            
            if success:
                logger.info(f"Plugin reloaded successfully: {plugin_name}")
            else:
                logger.error(f"Failed to reload plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin not found: {plugin_name}")
                return False
            
            logger.info(f"Unloading plugin: {plugin_name}")
            
            # Stop plugin if running
            plugin = self.plugins[plugin_name]
            if plugin.status.state.value == 'running':
                plugin.stop()
            
            # Cleanup plugin
            plugin.cleanup()
            
            # Remove from storage
            self.plugins.pop(plugin_name)
            
            # Update plugin lists
            for plugin_list in [self._loaded_plugins, self._enabled_plugins, 
                               self._disabled_plugins, self._error_plugins]:
                if plugin_name in plugin_list:
                    plugin_list.remove(plugin_name)
            
            logger.info(f"Plugin unloaded: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a new plugin"""
        try:
            plugin_path = Path(plugin_path)
            
            if not plugin_path.exists():
                logger.error(f"Plugin path not found: {plugin_path}")
                return False
            
            # Determine installation method
            if plugin_path.is_dir():
                return self._install_directory_plugin(plugin_path)
            elif plugin_path.suffix == '.zip':
                return self._install_zip_plugin(plugin_path)
            elif plugin_path.suffix == '.py':
                return self._install_file_plugin(plugin_path)
            else:
                logger.error(f"Unsupported plugin format: {plugin_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing plugin: {e}")
            return False
    
    def _install_directory_plugin(self, plugin_dir: Path) -> bool:
        """Install plugin from directory"""
        try:
            # Copy to plugins directory
            target_dir = self.plugins_path / plugin_dir.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(plugin_dir, target_dir)
            
            # Rediscover plugins
            self._discover_plugins()
            
            logger.info(f"Directory plugin installed: {plugin_dir.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing directory plugin: {e}")
            return False
    
    def _install_zip_plugin(self, zip_file: Path) -> bool:
        """Install plugin from ZIP file"""
        try:
            # Extract to plugins directory
            target_dir = self.plugins_path / zip_file.stem
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Rediscover plugins
            self._discover_plugins()
            
            logger.info(f"ZIP plugin installed: {zip_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing ZIP plugin: {e}")
            return False
    
    def _install_file_plugin(self, plugin_file: Path) -> bool:
        """Install plugin from single file"""
        try:
            # Copy to plugins directory
            target_file = self.plugins_path / plugin_file.name
            shutil.copy2(plugin_file, target_file)
            
            # Rediscover plugins
            self._discover_plugins()
            
            logger.info(f"File plugin installed: {plugin_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing file plugin: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a plugin"""
        return self.plugin_info.get(plugin_name)
    
    def get_all_plugins_info(self) -> Dict[str, PluginInfo]:
        """Get information about all plugins"""
        return self.plugin_info.copy()
    
    def get_plugin_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a plugin"""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].get_info()
        return None
    
    def get_plugins_by_state(self, state: str) -> List[str]:
        """Get plugins by state"""
        if state == 'loaded':
            return self._loaded_plugins.copy()
        elif state == 'enabled':
            return self._enabled_plugins.copy()
        elif state == 'disabled':
            return self._disabled_plugins.copy()
        elif state == 'error':
            return self._error_plugins.copy()
        else:
            return []
    
    def start_plugin_watcher(self):
        """Start plugin file watcher for hot-reloading"""
        if self._watch_enabled:
            return
        
        try:
            self._watch_enabled = True
            self._plugin_watcher = threading.Thread(target=self._watch_plugins, daemon=True)
            self._plugin_watcher.start()
            logger.info("Plugin watcher started")
            
        except Exception as e:
            logger.error(f"Error starting plugin watcher: {e}")
    
    def stop_plugin_watcher(self):
        """Stop plugin file watcher"""
        self._watch_enabled = False
        if self._plugin_watcher:
            self._plugin_watcher.join(timeout=1.0)
        logger.info("Plugin watcher stopped")
    
    def _watch_plugins(self):
        """Watch for plugin file changes"""
        # This is a simplified file watcher
        # In production, you might want to use watchdog or similar
        while self._watch_enabled:
            try:
                # Check for new plugins
                self._discover_plugins()
                
                # Sleep for a while
                import time
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in plugin watcher: {e}")
                import time
                time.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get plugin system status"""
        return {
            'plugin_manager': {
                'total_plugins': len(self._plugin_metadata),
                'loaded_plugins': len(self._loaded_plugins),
                'enabled_plugins': len(self._enabled_plugins),
                'disabled_plugins': len(self._disabled_plugins),
                'error_plugins': len(self._error_plugins),
                'watch_enabled': self._watch_enabled
            },
            'plugins': {
                'loaded': self._loaded_plugins,
                'enabled': self._enabled_plugins,
                'disabled': self._disabled_plugins,
                'error': self._error_plugins
            },
            'plugin_info': {
                name: {
                    'name': info.name,
                    'version': info.version,
                    'description': info.description,
                    'author': info.author,
                    'enabled': info.enabled,
                    'error_count': info.error_count,
                    'last_error': info.last_error,
                    'loaded_at': info.loaded_at.isoformat()
                }
                for name, info in self.plugin_info.items()
            }
        }
    
    def shutdown(self):
        """Shutdown plugin manager"""
        try:
            logger.info("Shutting down plugin manager...")
            
            # Stop watcher
            self.stop_plugin_watcher()
            
            # Unload all plugins
            for plugin_name in list(self.plugins.keys()):
                self.unload_plugin(plugin_name)
            
            # Clear storage
            self.plugins.clear()
            self.plugin_info.clear()
            self.plugin_classes.clear()
            self._plugin_metadata.clear()
            
            # Clear lists
            self._loaded_plugins.clear()
            self._enabled_plugins.clear()
            self._disabled_plugins.clear()
            self._error_plugins.clear()
            
            logger.info("Plugin manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
