"""
Plugin System
=============

Extensible plugin architecture for autonomous agent.
"""

import asyncio
import logging
import importlib.util
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Available hook types."""
    PRE_TASK = "pre_task"  # Before task execution
    POST_TASK = "post_task"  # After task execution
    ON_ERROR = "on_error"  # On task error
    ON_METRIC = "on_metric"  # When metrics are collected
    ON_HEALTH_CHECK = "on_health_check"  # After health check
    ON_SCALE = "on_scale"  # When workers are scaled
    ON_STARTUP = "on_startup"  # Agent startup
    ON_SHUTDOWN = "on_shutdown"  # Agent shutdown


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str = ""
    enabled: bool = True
    hooks: List[str] = field(default_factory=list)


class Plugin:
    """Base class for plugins."""
    
    info: PluginInfo = PluginInfo(
        name="base_plugin",
        version="1.0.0",
        description="Base plugin class",
    )
    
    async def on_load(self):
        """Called when plugin is loaded."""
        pass
    
    async def on_unload(self):
        """Called when plugin is unloaded."""
        pass
    
    async def pre_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before task execution.
        
        Args:
            task: Task dictionary
            
        Returns:
            Modified task dictionary
        """
        return task
    
    async def post_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called after task execution.
        
        Args:
            task: Task dictionary
            result: Task result
            
        Returns:
            Modified result dictionary
        """
        return result
    
    async def on_error(self, task: Dict[str, Any], error: Exception):
        """
        Called on task error.
        
        Args:
            task: Task dictionary
            error: Exception that occurred
        """
        pass
    
    async def on_metric(self, metrics: Dict[str, Any]):
        """
        Called when metrics are collected.
        
        Args:
            metrics: Metrics dictionary
        """
        pass
    
    async def on_health_check(self, health: Dict[str, Any]):
        """
        Called after health check.
        
        Args:
            health: Health check results
        """
        pass
    
    async def on_scale(self, old_workers: int, new_workers: int):
        """
        Called when workers are scaled.
        
        Args:
            old_workers: Previous worker count
            new_workers: New worker count
        """
        pass
    
    async def on_startup(self):
        """Called on agent startup."""
        pass
    
    async def on_shutdown(self):
        """Called on agent shutdown."""
        pass


class PluginManager:
    """
    Manages plugins for the autonomous agent.
    
    Features:
    - Load plugins from files or directories
    - Hook-based extension points
    - Enable/disable plugins at runtime
    - Plugin lifecycle management
    """
    
    def __init__(self, plugin_dir: str = "plugins"):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = Path(plugin_dir)
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[HookType, List[Callable]] = {
            hook: [] for hook in HookType
        }
        
        logger.info(f"Initialized PluginManager (dir: {self.plugin_dir})")
    
    async def load_plugins(self):
        """Load all plugins from plugin directory."""
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created plugin directory: {self.plugin_dir}")
            return
        
        for plugin_path in self.plugin_dir.glob("*.py"):
            if plugin_path.name.startswith("_"):
                continue
            
            try:
                await self.load_plugin(plugin_path)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e}")
    
    async def load_plugin(self, plugin_path: Path) -> Optional[str]:
        """
        Load a plugin from file.
        
        Args:
            plugin_path: Path to plugin file
            
        Returns:
            Plugin name if loaded successfully
        """
        spec = importlib.util.spec_from_file_location(
            plugin_path.stem, 
            plugin_path
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load plugin spec: {plugin_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find Plugin subclass in module
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type) and 
                issubclass(attr, Plugin) and 
                attr is not Plugin
            ):
                plugin_class = attr
                break
        
        if plugin_class is None:
            raise ValueError(f"No Plugin subclass found in {plugin_path}")
        
        # Instantiate plugin
        plugin = plugin_class()
        plugin_name = plugin.info.name
        
        # Call on_load
        await plugin.on_load()
        
        # Register plugin
        self._plugins[plugin_name] = plugin
        
        # Register hooks
        self._register_plugin_hooks(plugin)
        
        logger.info(f"Loaded plugin: {plugin_name} v{plugin.info.version}")
        return plugin_name
    
    async def unload_plugin(self, plugin_name: str):
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        plugin = self._plugins[plugin_name]
        
        # Call on_unload
        await plugin.on_unload()
        
        # Unregister hooks
        self._unregister_plugin_hooks(plugin)
        
        # Remove plugin
        del self._plugins[plugin_name]
        
        logger.info(f"Unloaded plugin: {plugin_name}")
    
    def _register_plugin_hooks(self, plugin: Plugin):
        """Register all hooks for a plugin."""
        hook_methods = {
            HookType.PRE_TASK: plugin.pre_task,
            HookType.POST_TASK: plugin.post_task,
            HookType.ON_ERROR: plugin.on_error,
            HookType.ON_METRIC: plugin.on_metric,
            HookType.ON_HEALTH_CHECK: plugin.on_health_check,
            HookType.ON_SCALE: plugin.on_scale,
            HookType.ON_STARTUP: plugin.on_startup,
            HookType.ON_SHUTDOWN: plugin.on_shutdown,
        }
        
        for hook_type, method in hook_methods.items():
            self._hooks[hook_type].append(method)
    
    def _unregister_plugin_hooks(self, plugin: Plugin):
        """Unregister all hooks for a plugin."""
        for hook_type in HookType:
            self._hooks[hook_type] = [
                h for h in self._hooks[hook_type]
                if not hasattr(h, "__self__") or h.__self__ is not plugin
            ]
    
    async def trigger_hook(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """
        Trigger a hook.
        
        Args:
            hook_type: Type of hook to trigger
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of results from all hook handlers
        """
        results = []
        
        for handler in self._hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Hook {hook_type.value} handler error: {e}",
                    exc_info=True
                )
        
        return results
    
    def get_plugins(self) -> List[Dict[str, Any]]:
        """Get information about all loaded plugins."""
        return [
            {
                "name": p.info.name,
                "version": p.info.version,
                "description": p.info.description,
                "author": p.info.author,
                "enabled": p.info.enabled,
            }
            for p in self._plugins.values()
        ]
    
    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self._plugins


# Example plugin template
EXAMPLE_PLUGIN_TEMPLATE = '''
"""
Example Plugin
==============

This is an example plugin for the autonomous agent.
"""

from core.plugin_system import Plugin, PluginInfo


class ExamplePlugin(Plugin):
    """Example plugin that logs task execution."""
    
    info = PluginInfo(
        name="example_plugin",
        version="1.0.0",
        description="Example plugin that logs task execution",
        author="Your Name",
    )
    
    async def on_load(self):
        print(f"[{self.info.name}] Plugin loaded!")
    
    async def pre_task(self, task):
        print(f"[{self.info.name}] Starting task: {task.get('id')}")
        return task
    
    async def post_task(self, task, result):
        print(f"[{self.info.name}] Completed task: {task.get('id')}")
        return result
    
    async def on_error(self, task, error):
        print(f"[{self.info.name}] Task failed: {error}")
'''
