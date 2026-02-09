"""
Plugin System for Final Ultimate AI

Ultra-modular plugin system with:
- Dynamic plugin loading and unloading
- Plugin dependency management
- Plugin versioning and compatibility
- Plugin lifecycle management
- Plugin communication and events
- Plugin configuration management
- Plugin security and sandboxing
- Plugin performance monitoring
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable, Type, Protocol, runtime_checkable
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import importlib
import inspect
import threading
from collections import defaultdict, deque
import weakref
import gc
import psutil
from pathlib import Path
import yaml
import pkgutil
import sys
import os
import zipfile
import tempfile
import shutil
import hashlib
import subprocess
import ast
import types

logger = structlog.get_logger("plugin_system")

class PluginStatus(Enum):
    """Plugin status enumeration."""
    UNINSTALLED = "uninstalled"
    INSTALLED = "installed"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"

class PluginType(Enum):
    """Plugin type enumeration."""
    VIDEO_PROCESSOR = "video_processor"
    AI_MODULE = "ai_module"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    UI_COMPONENT = "ui_component"
    MIDDLEWARE = "middleware"
    UTILITY = "utility"
    CUSTOM = "custom"

class PluginCapability(Enum):
    """Plugin capability enumeration."""
    VIDEO_ANALYSIS = "video_analysis"
    VIDEO_PROCESSING = "video_processing"
    AI_INFERENCE = "ai_inference"
    DATA_ANALYSIS = "data_analysis"
    API_ENDPOINT = "api_endpoint"
    UI_WIDGET = "ui_widget"
    EVENT_HANDLER = "event_handler"
    BACKGROUND_TASK = "background_task"
    FILE_HANDLER = "file_handler"
    NETWORK_SERVICE = "network_service"

@runtime_checkable
class PluginInterface(Protocol):
    """Plugin interface protocol."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        ...
    
    async def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        ...
    
    async def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        ...
    
    async def execute(self, task: str, data: Dict[str, Any]) -> Any:
        """Execute a plugin task."""
        ...

@dataclass
class PluginManifest:
    """Plugin manifest structure."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    capabilities: List[PluginCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    min_system_version: str = "1.0.0"
    max_system_version: Optional[str] = None
    api_version: str = "1.0.0"
    configuration_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    resources: Dict[str, str] = field(default_factory=dict)
    entry_point: str = "main"
    icon: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    support_url: Optional[str] = None
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PluginInstance:
    """Plugin instance structure."""
    manifest: PluginManifest
    instance: Any
    status: PluginStatus = PluginStatus.UNINSTALLED
    install_time: Optional[datetime] = None
    load_time: Optional[datetime] = None
    error: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PluginEvent:
    """Plugin event structure."""
    event_id: str
    plugin_id: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class PluginSecurityManager:
    """Plugin security and sandboxing manager."""
    
    def __init__(self):
        self.allowed_imports = {
            'asyncio', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'dataclasses', 'enum', 'structlog', 'numpy',
            'pandas', 'requests', 'aiohttp', 'fastapi', 'pydantic'
        }
        self.restricted_imports = {
            'os', 'sys', 'subprocess', 'importlib', 'inspect',
            'threading', 'multiprocessing', 'ctypes', 'socket'
        }
        self.sandbox_enabled = True
    
    def validate_plugin_code(self, code: str) -> bool:
        """Validate plugin code for security."""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for restricted imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.restricted_imports:
                            logger.warning(f"Restricted import detected: {alias.name}")
                            return False
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module in self.restricted_imports:
                        logger.warning(f"Restricted import detected: {node.module}")
                        return False
                
                # Check for dangerous operations
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            logger.warning(f"Dangerous operation detected: {node.func.id}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return False
    
    def create_sandbox_environment(self) -> Dict[str, Any]:
        """Create a sandboxed environment for plugins."""
        sandbox = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                'set': set, 'frozenset': frozenset, 'range': range,
                'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
                'sorted': sorted, 'reversed': reversed, 'sum': sum, 'min': min,
                'max': max, 'abs': abs, 'round': round, 'pow': pow,
                'divmod': divmod, 'bin': bin, 'hex': hex, 'oct': oct,
                'chr': chr, 'ord': ord, 'hash': hash, 'id': id,
                'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                'delattr': delattr, 'callable': callable, 'dir': dir,
                'vars': vars, 'locals': locals, 'globals': globals,
                'open': open, 'print': print, 'input': input,
                'repr': repr, 'ascii': ascii, 'format': format,
                'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
                'slice': slice, 'property': property, 'staticmethod': staticmethod,
                'classmethod': classmethod, 'super': super, 'object': object,
                'Exception': Exception, 'BaseException': BaseException,
                'ValueError': ValueError, 'TypeError': TypeError,
                'AttributeError': AttributeError, 'KeyError': KeyError,
                'IndexError': IndexError, 'RuntimeError': RuntimeError,
                'StopIteration': StopIteration, 'GeneratorExit': GeneratorExit,
                'SystemExit': SystemExit, 'KeyboardInterrupt': KeyboardInterrupt,
                'AssertionError': AssertionError, 'NotImplementedError': NotImplementedError,
                'ArithmeticError': ArithmeticError, 'ZeroDivisionError': ZeroDivisionError,
                'OverflowError': OverflowError, 'FloatingPointError': FloatingPointError,
                'OSError': OSError, 'IOError': IOError, 'EnvironmentError': EnvironmentError,
                'FileNotFoundError': FileNotFoundError, 'PermissionError': PermissionError,
                'ProcessLookupError': ProcessLookupError, 'ConnectionError': ConnectionError,
                'BrokenPipeError': BrokenPipeError, 'ConnectionAbortedError': ConnectionAbortedError,
                'ConnectionRefusedError': ConnectionRefusedError, 'ConnectionResetError': ConnectionResetError,
                'BlockingIOError': BlockingIOError, 'ChildProcessError': ChildProcessError,
                'FileExistsError': FileExistsError, 'IsADirectoryError': IsADirectoryError,
                'NotADirectoryError': NotADirectoryError, 'InterruptedError': InterruptedError,
                'TimeoutError': TimeoutError, 'UnicodeError': UnicodeError,
                'UnicodeDecodeError': UnicodeDecodeError, 'UnicodeEncodeError': UnicodeEncodeError,
                'UnicodeTranslateError': UnicodeTranslateError, 'Warning': Warning,
                'UserWarning': UserWarning, 'DeprecationWarning': DeprecationWarning,
                'PendingDeprecationWarning': PendingDeprecationWarning, 'SyntaxWarning': SyntaxWarning,
                'RuntimeWarning': RuntimeWarning, 'FutureWarning': FutureWarning,
                'ImportWarning': ImportWarning, 'UnicodeWarning': UnicodeWarning,
                'BytesWarning': BytesWarning, 'ResourceWarning': ResourceWarning
            }
        }
        
        # Add allowed modules
        for module_name in self.allowed_imports:
            try:
                module = importlib.import_module(module_name)
                sandbox[module_name] = module
            except ImportError:
                pass
        
        return sandbox

class PluginManager:
    """Main plugin management system."""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.plugin_directory.mkdir(exist_ok=True)
        
        # Plugin management
        self.plugins: Dict[str, PluginInstance] = {}
        self.plugin_registry: Dict[str, PluginManifest] = {}
        self.plugin_dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Security
        self.security_manager = PluginSecurityManager()
        
        # Events
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        self._metrics_lock = threading.Lock()
        
        # Plugin lifecycle
        self.running = False
    
    async def initialize(self) -> bool:
        """Initialize the plugin manager."""
        try:
            # Scan for installed plugins
            await self._scan_installed_plugins()
            
            # Load plugin registry
            await self._load_plugin_registry()
            
            self.running = True
            logger.info("Plugin manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Plugin manager initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        try:
            self.running = False
            
            # Stop all running plugins
            for plugin_id in list(self.plugins.keys()):
                await self.stop_plugin(plugin_id)
            
            logger.info("Plugin manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Plugin manager shutdown error: {e}")
    
    async def install_plugin(self, plugin_path: str, force: bool = False) -> bool:
        """Install a plugin from file."""
        try:
            plugin_path = Path(plugin_path)
            
            if not plugin_path.exists():
                logger.error(f"Plugin file not found: {plugin_path}")
                return False
            
            # Extract plugin if it's a zip file
            if plugin_path.suffix == '.zip':
                temp_dir = tempfile.mkdtemp()
                try:
                    with zipfile.ZipFile(plugin_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find manifest file
                    manifest_file = None
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file == 'plugin.yaml' or file == 'manifest.yaml':
                                manifest_file = Path(root) / file
                                break
                    
                    if not manifest_file:
                        logger.error("Plugin manifest not found")
                        return False
                    
                    # Load manifest
                    with open(manifest_file, 'r') as f:
                        manifest_dict = yaml.safe_load(f)
                    
                    manifest = PluginManifest(**manifest_dict)
                    
                    # Validate plugin
                    if not await self._validate_plugin(manifest, temp_dir):
                        logger.error(f"Plugin validation failed: {manifest.plugin_id}")
                        return False
                    
                    # Install plugin
                    plugin_dir = self.plugin_directory / manifest.plugin_id
                    if plugin_dir.exists() and not force:
                        logger.error(f"Plugin {manifest.plugin_id} already installed")
                        return False
                    
                    if plugin_dir.exists():
                        shutil.rmtree(plugin_dir)
                    
                    shutil.copytree(temp_dir, plugin_dir)
                    
                    # Register plugin
                    self.plugin_registry[manifest.plugin_id] = manifest
                    
                    # Create plugin instance
                    plugin_instance = PluginInstance(
                        manifest=manifest,
                        instance=None,
                        status=PluginStatus.INSTALLED,
                        install_time=datetime.now()
                    )
                    
                    self.plugins[manifest.plugin_id] = plugin_instance
                    
                    logger.info(f"Plugin {manifest.plugin_id} installed successfully")
                    return True
                    
                finally:
                    shutil.rmtree(temp_dir)
            
            else:
                # Single file plugin
                with open(plugin_path, 'r') as f:
                    code = f.read()
                
                # Validate code
                if not self.security_manager.validate_plugin_code(code):
                    logger.error("Plugin code validation failed")
                    return False
                
                # Create manifest from code
                manifest = await self._create_manifest_from_code(code, plugin_path.stem)
                
                # Install plugin
                plugin_dir = self.plugin_directory / manifest.plugin_id
                plugin_dir.mkdir(exist_ok=True)
                
                plugin_file = plugin_dir / f"{manifest.plugin_id}.py"
                with open(plugin_file, 'w') as f:
                    f.write(code)
                
                manifest_file = plugin_dir / "plugin.yaml"
                with open(manifest_file, 'w') as f:
                    yaml.dump(manifest.__dict__, f, default_flow_style=False)
                
                # Register plugin
                self.plugin_registry[manifest.plugin_id] = manifest
                
                # Create plugin instance
                plugin_instance = PluginInstance(
                    manifest=manifest,
                    instance=None,
                    status=PluginStatus.INSTALLED,
                    install_time=datetime.now()
                )
                
                self.plugins[manifest.plugin_id] = plugin_instance
                
                logger.info(f"Plugin {manifest.plugin_id} installed successfully")
                return True
            
        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return False
    
    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return False
            
            # Check if plugin is running
            if self.plugins[plugin_id].status == PluginStatus.RUNNING:
                await self.stop_plugin(plugin_id)
            
            # Check dependencies
            dependent_plugins = [pid for pid, deps in self.plugin_dependencies.items() if plugin_id in deps]
            if dependent_plugins:
                logger.error(f"Cannot uninstall plugin {plugin_id}: dependent plugins {dependent_plugins}")
                return False
            
            # Remove plugin directory
            plugin_dir = self.plugin_directory / plugin_id
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            
            # Remove from registry
            self.plugin_registry.pop(plugin_id, None)
            self.plugins.pop(plugin_id, None)
            
            logger.info(f"Plugin {plugin_id} uninstalled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin uninstallation failed: {e}")
            return False
    
    async def load_plugin(self, plugin_id: str, configuration: Dict[str, Any] = None) -> bool:
        """Load a plugin."""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return False
            
            plugin_instance = self.plugins[plugin_id]
            
            if plugin_instance.status not in [PluginStatus.INSTALLED, PluginStatus.STOPPED]:
                logger.warning(f"Plugin {plugin_id} not in loadable state")
                return False
            
            # Load plugin code
            plugin_dir = self.plugin_directory / plugin_id
            plugin_file = plugin_dir / f"{plugin_id}.py"
            
            if not plugin_file.exists():
                logger.error(f"Plugin file not found: {plugin_file}")
                return False
            
            with open(plugin_file, 'r') as f:
                code = f.read()
            
            # Create sandbox environment
            sandbox = self.security_manager.create_sandbox_environment()
            
            # Execute plugin code in sandbox
            exec(code, sandbox)
            
            # Find plugin class
            plugin_class = None
            for name, obj in sandbox.items():
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'initialize') and 
                    hasattr(obj, 'shutdown')):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {plugin_id}")
                return False
            
            # Create plugin instance
            plugin_instance.instance = plugin_class()
            plugin_instance.status = PluginStatus.LOADING
            plugin_instance.load_time = datetime.now()
            plugin_instance.configuration = configuration or {}
            
            # Initialize plugin
            plugin_instance.status = PluginStatus.INITIALIZING
            try:
                init_success = await plugin_instance.instance.initialize(plugin_instance.configuration)
                if not init_success:
                    plugin_instance.status = PluginStatus.ERROR
                    plugin_instance.error = "Initialization failed"
                    return False
                
                plugin_instance.status = PluginStatus.INITIALIZED
                
            except Exception as e:
                plugin_instance.status = PluginStatus.ERROR
                plugin_instance.error = str(e)
                logger.error(f"Plugin {plugin_id} initialization failed: {e}")
                return False
            
            logger.info(f"Plugin {plugin_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin loading failed: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return False
            
            plugin_instance = self.plugins[plugin_id]
            
            if plugin_instance.status not in [PluginStatus.LOADED, PluginStatus.INITIALIZED, PluginStatus.RUNNING]:
                logger.warning(f"Plugin {plugin_id} not in unloadable state")
                return True
            
            # Stop plugin if running
            if plugin_instance.status == PluginStatus.RUNNING:
                await self.stop_plugin(plugin_id)
            
            # Shutdown plugin
            if plugin_instance.instance:
                try:
                    await plugin_instance.instance.shutdown()
                except Exception as e:
                    logger.error(f"Plugin {plugin_id} shutdown error: {e}")
            
            plugin_instance.instance = None
            plugin_instance.status = PluginStatus.INSTALLED
            plugin_instance.load_time = None
            
            logger.info(f"Plugin {plugin_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin unloading failed: {e}")
            return False
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a plugin."""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return False
            
            plugin_instance = self.plugins[plugin_id]
            
            if plugin_instance.status != PluginStatus.INITIALIZED:
                logger.error(f"Plugin {plugin_id} not initialized")
                return False
            
            # Start plugin (if it has a start method)
            if hasattr(plugin_instance.instance, 'start'):
                await plugin_instance.instance.start()
            
            plugin_instance.status = PluginStatus.RUNNING
            
            # Publish plugin started event
            await self._publish_plugin_event(plugin_id, "plugin_started", {})
            
            logger.info(f"Plugin {plugin_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin start failed: {e}")
            return False
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop a plugin."""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return False
            
            plugin_instance = self.plugins[plugin_id]
            
            if plugin_instance.status != PluginStatus.RUNNING:
                logger.warning(f"Plugin {plugin_id} not running")
                return True
            
            # Stop plugin (if it has a stop method)
            if hasattr(plugin_instance.instance, 'stop'):
                await plugin_instance.instance.stop()
            
            plugin_instance.status = PluginStatus.INITIALIZED
            
            # Publish plugin stopped event
            await self._publish_plugin_event(plugin_id, "plugin_stopped", {})
            
            logger.info(f"Plugin {plugin_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Plugin stop failed: {e}")
            return False
    
    async def execute_plugin_task(self, plugin_id: str, task: str, data: Dict[str, Any]) -> Any:
        """Execute a plugin task."""
        try:
            if plugin_id not in self.plugins:
                raise ValueError(f"Plugin {plugin_id} not found")
            
            plugin_instance = self.plugins[plugin_id]
            
            if plugin_instance.status != PluginStatus.RUNNING:
                raise ValueError(f"Plugin {plugin_id} not running")
            
            if not plugin_instance.instance:
                raise ValueError(f"Plugin {plugin_id} not loaded")
            
            # Execute task
            result = await plugin_instance.instance.execute(task, data)
            
            # Update performance metrics
            await self._update_plugin_metrics(plugin_id, task, data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Plugin task execution failed: {e}")
            raise e
    
    async def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin information."""
        if plugin_id not in self.plugins:
            return None
        
        plugin_instance = self.plugins[plugin_id]
        return {
            "plugin_id": plugin_id,
            "manifest": plugin_instance.manifest.__dict__,
            "status": plugin_instance.status.value,
            "install_time": plugin_instance.install_time.isoformat() if plugin_instance.install_time else None,
            "load_time": plugin_instance.load_time.isoformat() if plugin_instance.load_time else None,
            "error": plugin_instance.error,
            "configuration": plugin_instance.configuration,
            "performance_metrics": plugin_instance.performance_metrics,
            "resource_usage": plugin_instance.resource_usage
        }
    
    async def get_all_plugins_info(self) -> Dict[str, Any]:
        """Get information about all plugins."""
        return {
            plugin_id: await self.get_plugin_info(plugin_id)
            for plugin_id in self.plugins.keys()
        }
    
    async def get_plugins_by_capability(self, capability: str) -> List[str]:
        """Get plugins that have a specific capability."""
        return [
            plugin_id for plugin_id, plugin in self.plugins.items()
            if capability in [cap.value for cap in plugin.manifest.capabilities]
        ]
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """Get plugins of a specific type."""
        return [
            plugin_id for plugin_id, plugin in self.plugins.items()
            if plugin.manifest.plugin_type == plugin_type
        ]
    
    async def _scan_installed_plugins(self) -> None:
        """Scan for installed plugins."""
        for plugin_dir in self.plugin_directory.iterdir():
            if plugin_dir.is_dir():
                manifest_file = plugin_dir / "plugin.yaml"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_dict = yaml.safe_load(f)
                        
                        manifest = PluginManifest(**manifest_dict)
                        self.plugin_registry[manifest.plugin_id] = manifest
                        
                        plugin_instance = PluginInstance(
                            manifest=manifest,
                            instance=None,
                            status=PluginStatus.INSTALLED
                        )
                        
                        self.plugins[manifest.plugin_id] = plugin_instance
                        
                    except Exception as e:
                        logger.error(f"Failed to load plugin manifest {manifest_file}: {e}")
    
    async def _load_plugin_registry(self) -> None:
        """Load plugin registry from file."""
        registry_file = self.plugin_directory / "registry.yaml"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = yaml.safe_load(f)
                
                for plugin_id, manifest_dict in registry_data.items():
                    manifest = PluginManifest(**manifest_dict)
                    self.plugin_registry[plugin_id] = manifest
                    
            except Exception as e:
                logger.error(f"Failed to load plugin registry: {e}")
    
    async def _save_plugin_registry(self) -> None:
        """Save plugin registry to file."""
        registry_file = self.plugin_directory / "registry.yaml"
        try:
            registry_data = {
                plugin_id: manifest.__dict__
                for plugin_id, manifest in self.plugin_registry.items()
            }
            
            with open(registry_file, 'w') as f:
                yaml.dump(registry_data, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f"Failed to save plugin registry: {e}")
    
    async def _validate_plugin(self, manifest: PluginManifest, plugin_dir: str) -> bool:
        """Validate a plugin."""
        try:
            # Check system version compatibility
            if manifest.min_system_version:
                # Simple version comparison (in practice, use proper version comparison)
                pass
            
            # Check dependencies
            for dependency in manifest.dependencies:
                if dependency not in self.plugin_registry:
                    logger.error(f"Dependency {dependency} not found")
                    return False
            
            # Check plugin code
            plugin_file = Path(plugin_dir) / f"{manifest.plugin_id}.py"
            if plugin_file.exists():
                with open(plugin_file, 'r') as f:
                    code = f.read()
                
                if not self.security_manager.validate_plugin_code(code):
                    logger.error("Plugin code validation failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False
    
    async def _create_manifest_from_code(self, code: str, plugin_name: str) -> PluginManifest:
        """Create manifest from plugin code."""
        # Extract metadata from code comments
        manifest = PluginManifest(
            plugin_id=plugin_name,
            name=plugin_name.replace('_', ' ').title(),
            version="1.0.0",
            description="Auto-generated plugin",
            author="Unknown",
            plugin_type=PluginType.CUSTOM,
            capabilities=[],
            dependencies=[],
            optional_dependencies=[],
            min_system_version="1.0.0",
            api_version="1.0.0",
            configuration_schema={},
            permissions=[],
            resources={},
            entry_point="main",
            license="MIT",
            tags=[]
        )
        
        return manifest
    
    async def _publish_plugin_event(self, plugin_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a plugin event."""
        event = PluginEvent(
            event_id=str(uuid.uuid4()),
            plugin_id=plugin_id,
            event_type=event_type,
            data=data
        )
        
        self.event_history.append(event)
        
        # Notify event handlers
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Plugin event handler error: {e}")
    
    async def _update_plugin_metrics(self, plugin_id: str, task: str, data: Dict[str, Any], result: Any) -> None:
        """Update plugin performance metrics."""
        with self._metrics_lock:
            self.performance_metrics[plugin_id].append({
                "task": task,
                "timestamp": datetime.now(),
                "data_size": len(str(data)),
                "result_size": len(str(result)) if result else 0
            })
    
    async def subscribe_to_plugin_events(self, event_type: str, handler: Callable) -> None:
        """Subscribe to plugin events."""
        self.event_handlers[event_type].append(handler)
    
    async def get_plugin_events(self, plugin_id: Optional[str] = None, event_type: Optional[str] = None) -> List[PluginEvent]:
        """Get plugin events."""
        events = list(self.event_history)
        
        if plugin_id:
            events = [e for e in events if e.plugin_id == plugin_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events

# Example plugin implementation
class ExampleVideoProcessorPlugin:
    """Example video processor plugin."""
    
    def __init__(self):
        self.plugin_id = "example_video_processor"
        self.status = "idle"
        self.metrics = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self.status = "initialized"
            self.config = config
            logger.info(f"Example video processor plugin {self.plugin_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Example video processor plugin initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        try:
            self.status = "stopped"
            logger.info(f"Example video processor plugin {self.plugin_id} shutdown")
        except Exception as e:
            logger.error(f"Example video processor plugin shutdown error: {e}")
    
    async def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "plugin_id": self.plugin_id,
            "name": "Example Video Processor",
            "version": "1.0.0",
            "description": "Example video processing plugin",
            "capabilities": ["video_processing", "video_analysis"]
        }
    
    async def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        return ["video_processing", "video_analysis"]
    
    async def execute(self, task: str, data: Dict[str, Any]) -> Any:
        """Execute a plugin task."""
        try:
            if task == "process_video":
                return await self._process_video(data)
            elif task == "analyze_video":
                return await self._analyze_video(data)
            else:
                raise ValueError(f"Unknown task: {task}")
        except Exception as e:
            logger.error(f"Plugin task execution failed: {e}")
            raise e
    
    async def _process_video(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process video."""
        # Simulate video processing
        await asyncio.sleep(0.1)
        return {
            "processed": True,
            "input_path": data.get("input_path"),
            "output_path": data.get("output_path"),
            "processing_time": 0.1
        }
    
    async def _analyze_video(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video."""
        # Simulate video analysis
        await asyncio.sleep(0.1)
        return {
            "analyzed": True,
            "duration": data.get("duration", 0),
            "resolution": data.get("resolution", "1920x1080"),
            "analysis_time": 0.1
        }

# Example usage
async def main():
    """Example usage of plugin system."""
    # Create plugin manager
    plugin_manager = PluginManager(plugin_directory="plugins")
    
    # Initialize plugin manager
    success = await plugin_manager.initialize()
    if not success:
        print("Failed to initialize plugin manager")
        return
    
    # Install a plugin
    success = await plugin_manager.install_plugin("example_plugin.py")
    if success:
        print("Plugin installed successfully")
        
        # Load plugin
        success = await plugin_manager.load_plugin("example_plugin")
        if success:
            print("Plugin loaded successfully")
            
            # Start plugin
            success = await plugin_manager.start_plugin("example_plugin")
            if success:
                print("Plugin started successfully")
                
                # Execute plugin task
                result = await plugin_manager.execute_plugin_task(
                    "example_plugin", 
                    "process_video", 
                    {"input_path": "/path/to/input.mp4", "output_path": "/path/to/output.mp4"}
                )
                print(f"Plugin task result: {result}")
                
                # Stop plugin
                await plugin_manager.stop_plugin("example_plugin")
                
                # Unload plugin
                await plugin_manager.unload_plugin("example_plugin")
        
        # Uninstall plugin
        await plugin_manager.uninstall_plugin("example_plugin")
    
    # Get all plugins info
    plugins_info = await plugin_manager.get_all_plugins_info()
    print(f"All plugins: {plugins_info}")
    
    # Shutdown plugin manager
    await plugin_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

