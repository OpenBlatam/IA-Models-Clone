from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import importlib
import importlib.util
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from ..core.exceptions import PluginError, ValidationError, DependencyError
from ..core.types import PluginInfo
from .base import BasePlugin
from typing import Any, List, Dict, Optional
"""
Plugin Loader - Production-Ready Plugin Loading System

This module provides robust plugin loading capabilities with:
- Dynamic discovery and loading
- Comprehensive validation
- Error handling and recovery
- Performance monitoring
- User-friendly error messages
"""



logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of plugin loading operation."""
    success: bool
    plugin: Optional[BasePlugin] = None
    error: Optional[str] = None
    load_time: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self) -> Any:
        if self.warnings is None:
            self.warnings = []


class PluginLoader:
    """
    Production-ready plugin loader with comprehensive error handling.
    
    Features:
    - Dynamic plugin discovery
    - Robust error handling and recovery
    - Performance monitoring
    - Detailed logging
    - User-friendly error messages
    """
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        
    """__init__ function."""
self.max_retries = max_retries
        self.timeout = timeout
        self.loaded_modules: Dict[str, Any] = {}
        self.load_stats = {
            'total_attempts': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'total_load_time': 0.0
        }
        
        logger.info(f"PluginLoader initialized (max_retries={max_retries}, timeout={timeout}s)")
    
    async def load_plugin(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]] = None,
        retry_on_failure: bool = True
    ) -> LoadResult:
        """
        Load a plugin with comprehensive error handling.
        
        Args:
            plugin_name: Name or path of the plugin to load
            config: Plugin configuration
            retry_on_failure: Whether to retry on failure
            
        Returns:
            LoadResult with success status and details
        """
        start_time = time.time()
        self.load_stats['total_attempts'] += 1
        
        logger.info(f"Loading plugin: {plugin_name}")
        
        try:
            # Try to load the plugin
            plugin = await self._load_plugin_with_retry(plugin_name, config, retry_on_failure)
            
            if plugin:
                load_time = time.time() - start_time
                self.load_stats['successful_loads'] += 1
                self.load_stats['total_load_time'] += load_time
                
                logger.info(f"✅ Plugin '{plugin_name}' loaded successfully in {load_time:.2f}s")
                
                return LoadResult(
                    success=True,
                    plugin=plugin,
                    load_time=load_time
                )
            else:
                raise PluginError(f"Failed to load plugin: {plugin_name}")
                
        except Exception as e:
            load_time = time.time() - start_time
            self.load_stats['failed_loads'] += 1
            
            error_msg = self._format_error_message(plugin_name, e)
            logger.error(f"❌ Failed to load plugin '{plugin_name}': {error_msg}")
            
            return LoadResult(
                success=False,
                error=error_msg,
                load_time=load_time
            )
    
    async def _load_plugin_with_retry(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]], 
        retry_on_failure: bool
    ) -> Optional[BasePlugin]:
        """Load plugin with retry logic."""
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Try to load from different sources
                plugin = await self._try_load_from_source(plugin_name, config)
                if plugin:
                    return plugin
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and retry_on_failure:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed for '{plugin_name}'. "
                        f"Retrying in {wait_time}s... Error: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        if last_error:
            raise last_error
        
        return None
    
    async def _try_load_from_source(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]]
    ) -> Optional[BasePlugin]:
        """Try to load plugin from different sources."""
        
        # Try as a module path first
        if '.' in plugin_name:
            return await self._load_from_module_path(plugin_name, config)
        
        # Try as a file path
        if Path(plugin_name).exists():
            return await self._load_from_file_path(plugin_name, config)
        
        # Try as a plugin name in common directories
        return await self._load_from_plugin_name(plugin_name, config)
    
    async def _load_from_module_path(
        self, 
        module_path: str, 
        config: Optional[Dict[str, Any]]
    ) -> Optional[BasePlugin]:
        """Load plugin from module path (e.g., 'my_package.my_plugin')."""
        try:
            module = importlib.import_module(module_path)
            return await self._find_and_instantiate_plugin(module, config)
        except ImportError as e:
            raise PluginError(f"Module '{module_path}' not found: {e}")
    
    async def _load_from_file_path(
        self, 
        file_path: str, 
        config: Optional[Dict[str, Any]]
    ) -> Optional[BasePlugin]:
        """Load plugin from file path."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise PluginError(f"Plugin file not found: {file_path}")
            
            # Load the module from file
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            
            if spec is None or spec.loader is None:
                raise PluginError(f"Invalid plugin file: {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return await self._find_and_instantiate_plugin(module, config)
            
        except Exception as e:
            raise PluginError(f"Failed to load plugin from file '{file_path}': {e}")
    
    async def _load_from_plugin_name(
        self, 
        plugin_name: str, 
        config: Optional[Dict[str, Any]]
    ) -> Optional[BasePlugin]:
        """Load plugin by searching common plugin directories."""
        plugin_dirs = [
            "./plugins",
            "./ai_video/plugins", 
            "./extensions",
            "~/.ai_video/plugins"
        ]
        
        for plugin_dir in plugin_dirs:
            path = Path(plugin_dir).expanduser()
            if not path.exists():
                continue
            
            # Look for plugin files
            plugin_file = await self._find_plugin_file(path, plugin_name)
            if plugin_file:
                return await self._load_from_file_path(str(plugin_file), config)
        
        raise PluginError(f"Plugin '{plugin_name}' not found in any plugin directory")
    
    async def _find_plugin_file(self, plugin_dir: Path, plugin_name: str) -> Optional[Path]:
        """Find plugin file in directory."""
        # Look for exact match first
        for pattern in [
            f"{plugin_name}.py",
            f"{plugin_name}/__init__.py",
            f"{plugin_name}/plugin.py",
            f"{plugin_name}/main.py"
        ]:
            file_path = plugin_dir / pattern
            if file_path.exists():
                return file_path
        
        # Look for files containing the plugin name
        for file_path in plugin_dir.rglob("*.py"):
            if not file_path.name.startswith('_'):
                # Check if file contains plugin class
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        if f"class {plugin_name}" in content or f"name = '{plugin_name}'" in content:
                            return file_path
                except Exception:
                    continue
        
        return None
    
    async def _find_and_instantiate_plugin(
        self, 
        module: Any, 
        config: Optional[Dict[str, Any]]
    ) -> Optional[BasePlugin]:
        """Find plugin class in module and instantiate it."""
        plugin_classes = []
        
        # Find all plugin classes in the module
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BasePlugin) and 
                obj != BasePlugin):
                plugin_classes.append(obj)
        
        if not plugin_classes:
            raise PluginError(f"No plugin classes found in module {module.__name__}")
        
        if len(plugin_classes) > 1:
            logger.warning(f"Multiple plugin classes found in {module.__name__}, using first one")
        
        # Instantiate the first plugin class
        plugin_class = plugin_classes[0]
        try:
            plugin = plugin_class(config or {})
            return plugin
        except Exception as e:
            raise PluginError(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
    
    def _format_error_message(self, plugin_name: str, error: Exception) -> str:
        """Format user-friendly error messages."""
        if isinstance(error, ImportError):
            return f"Plugin module '{plugin_name}' could not be imported. Please check if it's installed and accessible."
        
        elif isinstance(error, FileNotFoundError):
            return f"Plugin file '{plugin_name}' not found. Please check the file path."
        
        elif isinstance(error, ValidationError):
            return f"Plugin '{plugin_name}' has invalid configuration: {error.message}"
        
        elif isinstance(error, DependencyError):
            return f"Plugin '{plugin_name}' has missing dependencies: {error.message}"
        
        elif isinstance(error, PluginError):
            return f"Plugin '{plugin_name}' error: {error.message}"
        
        else:
            return f"Unexpected error loading plugin '{plugin_name}': {str(error)}"
    
    async def discover_plugins(self, plugin_dirs: Optional[List[str]] = None) -> List[PluginInfo]:
        """
        Discover all available plugins in specified directories.
        
        Args:
            plugin_dirs: List of directories to search (uses defaults if None)
            
        Returns:
            List of discovered plugin information
        """
        if plugin_dirs is None:
            plugin_dirs = [
                "./plugins",
                "./ai_video/plugins",
                "./extensions",
                "~/.ai_video/plugins"
            ]
        
        discovered_plugins = []
        
        for plugin_dir in plugin_dirs:
            path = Path(plugin_dir).expanduser()
            if not path.exists():
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue
            
            logger.info(f"🔍 Discovering plugins in: {plugin_dir}")
            
            try:
                plugins = await self._discover_plugins_in_directory(path)
                discovered_plugins.extend(plugins)
                logger.info(f"✅ Found {len(plugins)} plugins in {plugin_dir}")
                
            except Exception as e:
                logger.error(f"❌ Error discovering plugins in {plugin_dir}: {e}")
        
        logger.info(f"🎯 Total plugins discovered: {len(discovered_plugins)}")
        return discovered_plugins
    
    async def _discover_plugins_in_directory(self, plugin_dir: Path) -> List[PluginInfo]:
        """Discover plugins in a specific directory."""
        plugins = []
        
        for item in plugin_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                plugin_info = await self._analyze_plugin_file(item)
                if plugin_info:
                    plugins.append(plugin_info)
            
            elif item.is_dir() and not item.name.startswith('_'):
                plugin_info = await self._analyze_plugin_directory(item)
                if plugin_info:
                    plugins.append(plugin_info)
        
        return plugins
    
    async def _analyze_plugin_file(self, file_path: Path) -> Optional[PluginInfo]:
        """Analyze a plugin file to extract metadata."""
        try:
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    return await self._extract_plugin_info(obj, file_path)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to analyze plugin file {file_path}: {e}")
            return None
    
    async def _analyze_plugin_directory(self, dir_path: Path) -> Optional[PluginInfo]:
        """Analyze a plugin directory to extract metadata."""
        # Look for main plugin files
        for main_file in ["__init__.py", "plugin.py", "main.py"]:
            main_path = dir_path / main_file
            if main_path.exists():
                return await self._analyze_plugin_file(main_path)
        
        return None
    
    async def _extract_plugin_info(self, plugin_class: Type[BasePlugin], file_path: Path) -> PluginInfo:
        """Extract plugin information from a plugin class."""
        try:
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            
            return PluginInfo(
                name=metadata.name,
                version=metadata.version,
                description=metadata.description,
                author=metadata.author,
                category=metadata.category,
                dependencies=metadata.dependencies,
                config_schema=metadata.config_schema,
                is_enabled=True,
                is_loaded=False
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract plugin info from {file_path}: {e}")
            return None
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        stats = self.load_stats.copy()
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_loads'] / stats['total_attempts']
            stats['avg_load_time'] = stats['total_load_time'] / stats['successful_loads']
        else:
            stats['success_rate'] = 0.0
            stats['avg_load_time'] = 0.0
        
        return stats
    
    @asynccontextmanager
    async def load_plugin_context(self, plugin_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Context manager for loading plugins with automatic cleanup.
        
        Usage:
            async with loader.load_plugin_context("my_plugin") as plugin:
                await plugin.do_something()
        """
        result = await self.load_plugin(plugin_name, config)
        
        if not result.success:
            raise PluginError(f"Failed to load plugin: {result.error}")
        
        try:
            yield result.plugin
        finally:
            # Cleanup if needed
            if hasattr(result.plugin, 'cleanup'):
                await result.plugin.cleanup() 