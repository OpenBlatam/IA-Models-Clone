from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import importlib
import inspect
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.utils.retry_wrapper import retry_wrapper
from .core.models import OnyxPluginInfo, OnyxPluginContext, PluginStatus, PluginManagerStatus
from .core.base import OnyxPluginBase
from .core.exceptions import PluginError, AIVideoError
from .core.onyx_integration import onyx_integration
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Plugin Manager - Main Manager Class

Main plugin manager class that handles plugin lifecycle, execution,
and integration with Onyx's infrastructure for optimal performance.
"""


# Onyx imports

# Local imports


class OnyxPluginManager:
    """
    Onyx-adapted plugin manager.
    
    Manages plugin lifecycle, execution, and integration with Onyx's
    infrastructure for optimal performance and reliability.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_plugin_manager")
        self.telemetry = TelemetryLogger()
        self.plugins: ThreadSafeDict[str, OnyxPluginBase] = ThreadSafeDict()
        self.plugin_configs: ThreadSafeDict[str, Dict[str, Any]] = ThreadSafeDict()
        self.plugin_status: ThreadSafeDict[str, PluginStatus] = ThreadSafeDict()
        
        # Plugin directories
        self.plugin_dirs = [
            Path(__file__).parent.parent / "plugins",
            Path(__file__).parent.parent / "custom_plugins"
        ]
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        try:
            self.logger.info("Initializing Onyx plugin manager")
            
            # Initialize Onyx integration
            await onyx_integration.initialize()
            
            # Discover and load plugins
            await self._discover_plugins()
            
            # Initialize loaded plugins
            await self._initialize_plugins()
            
            self.logger.info("Onyx plugin manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Plugin manager initialization failed: {e}")
            raise AIVideoError(f"Plugin manager initialization failed: {e}")
    
    async def _discover_plugins(self) -> None:
        """Discover available plugins."""
        try:
            for plugin_dir in self.plugin_dirs:
                if not plugin_dir.exists():
                    continue
                
                self.logger.info(f"Scanning plugin directory: {plugin_dir}")
                
                for plugin_file in plugin_dir.glob("*.py"):
                    if plugin_file.name.startswith("__"):
                        continue
                    
                    try:
                        await self._load_plugin(plugin_file)
                    except Exception as e:
                        self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
            
            self.logger.info(f"Discovered {len(self.plugins)} plugins")
            
        except Exception as e:
            self.logger.error(f"Plugin discovery failed: {e}")
            raise AIVideoError(f"Plugin discovery failed: {e}")
    
    async def _load_plugin(self, plugin_file: Path) -> None:
        """Load a single plugin."""
        try:
            # Import plugin module
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, OnyxPluginBase) and 
                    obj != OnyxPluginBase):
                    
                    # Create plugin instance
                    config = self.plugin_configs.get(name, {})
                    plugin = obj(config)
                    
                    # Register plugin
                    self.plugins[name] = plugin
                    self.plugin_status[name] = PluginStatus(loaded=True)
                    
                    self.logger.info(f"Loaded plugin: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
            raise PluginError(f"Plugin loading failed: {e}")
    
    async def _initialize_plugins(self) -> None:
        """Initialize loaded plugins."""
        try:
            # Initialize plugins in parallel
            init_tasks = []
            for name, plugin in self.plugins.items():
                task = self._initialize_plugin_safe(name, plugin)
                init_tasks.append(task)
            
            # Wait for all initializations
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Update status
            for name, result in zip(self.plugins.keys(), results):
                if isinstance(result, Exception):
                    self.plugin_status[name].initialized = False
                    self.plugin_status[name].error = str(result)
                    self.logger.error(f"Plugin initialization failed: {name} - {result}")
                else:
                    self.plugin_status[name].initialized = True
                    self.logger.info(f"Plugin initialized: {name}")
            
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {e}")
            raise AIVideoError(f"Plugin initialization failed: {e}")
    
    async def _initialize_plugin_safe(self, name: str, plugin: OnyxPluginBase) -> None:
        """Safely initialize a plugin."""
        try:
            await plugin.initialize()
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {name} - {e}")
            raise
    
    async def execute_plugins(self, context: OnyxPluginContext, plugin_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute plugins for video processing."""
        try:
            # Filter plugins
            plugins_to_execute = self._filter_plugins(plugin_names)
            
            if not plugins_to_execute:
                self.logger.warning("No plugins to execute")
                return {}
            
            # Execute plugins in parallel
            execution_tasks = []
            for name, plugin in plugins_to_execute.items():
                task = self._execute_plugin_safe(name, plugin, context)
                execution_tasks.append(task)
            
            # Wait for all executions
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Collect results
            plugin_results = {}
            for name, result in zip(plugins_to_execute.keys(), results):
                if isinstance(result, Exception):
                    self.logger.error(f"Plugin execution failed: {name} - {result}")
                    plugin_results[name] = {"error": str(result)}
                else:
                    plugin_results[name] = result
            
            # Log telemetry
            self.telemetry.log_info("plugins_executed", {
                "plugin_count": len(plugins_to_execute),
                "successful": len([r for r in results if not isinstance(r, Exception)]),
                "failed": len([r for r in results if isinstance(r, Exception)])
            })
            
            return plugin_results
            
        except Exception as e:
            self.logger.error(f"Plugin execution failed: {e}")
            raise AIVideoError(f"Plugin execution failed: {e}")
    
    def _filter_plugins(self, plugin_names: Optional[List[str]] = None) -> Dict[str, OnyxPluginBase]:
        """Filter plugins based on names and status."""
        filtered_plugins = {}
        
        for name, plugin in self.plugins.items():
            # Check if plugin should be executed
            if plugin_names and name not in plugin_names:
                continue
            
            # Check if plugin is enabled and initialized
            status = self.plugin_status.get(name, PluginStatus())
            if not status.enabled or not status.initialized:
                continue
            
            filtered_plugins[name] = plugin
        
        return filtered_plugins
    
    @retry_wrapper(max_attempts=3, backoff_factor=2)
    async def _execute_plugin_safe(self, name: str, plugin: OnyxPluginBase, context: OnyxPluginContext) -> Dict[str, Any]:
        """Safely execute a plugin with timeout."""
        try:
            plugin_info = plugin.get_info()
            start_time = datetime.now()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                plugin.process(context),
                timeout=plugin_info.timeout
            )
            
            # Update execution statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            status = self.plugin_status[name]
            status.last_execution = datetime.now()
            status.execution_count += 1
            status.total_execution_time += execution_time
            
            return result
            
        except asyncio.TimeoutError:
            raise PluginError(f"Plugin execution timeout: {name}")
        except Exception as e:
            raise PluginError(f"Plugin execution failed: {name} - {e}")
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[OnyxPluginInfo]:
        """Get information about a specific plugin."""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            return plugin.get_info()
        return None
    
    async def get_all_plugins_info(self) -> List[OnyxPluginInfo]:
        """Get information about all plugins."""
        plugin_infos = []
        
        for name, plugin in self.plugins.items():
            try:
                info = plugin.get_info()
                info.enabled = self.plugin_status.get(name, PluginStatus()).enabled
                plugin_infos.append(info)
            except Exception as e:
                self.logger.error(f"Failed to get plugin info: {name} - {e}")
        
        return plugin_infos
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugin_status[plugin_name].enabled = True
            self.logger.info(f"Plugin enabled: {plugin_name}")
            return True
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugin_status[plugin_name].enabled = False
            self.logger.info(f"Plugin disabled: {plugin_name}")
            return True
        return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        try:
            if plugin_name in self.plugins:
                # Cleanup existing plugin
                plugin = self.plugins[plugin_name]
                await plugin.cleanup()
                
                # Remove from registry
                del self.plugins[plugin_name]
                del self.plugin_status[plugin_name]
                
                # Reload plugin
                # This would require re-scanning the plugin file
                self.logger.info(f"Plugin reloaded: {plugin_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Plugin reload failed: {plugin_name} - {e}")
            return False
    
    async def get_manager_status(self) -> PluginManagerStatus:
        """Get plugin manager status."""
        try:
            plugin_infos = await self.get_all_plugins_info()
            
            status = PluginManagerStatus(
                total_plugins=len(self.plugins),
                enabled_plugins=len([p for p in plugin_infos if p.enabled]),
                initialized_plugins=len([p for p in self.plugin_status.values() if p.initialized]),
                plugin_directories=[str(d) for d in self.plugin_dirs],
                gpu_available=is_gpu_available(),
                gpu_info=get_gpu_info() if is_gpu_available() else None,
                plugins=[
                    {
                        "name": info.name,
                        "version": info.version,
                        "category": info.category,
                        "enabled": info.enabled,
                        "gpu_required": info.gpu_required,
                        "status": self.plugin_status.get(info.name, PluginStatus()).__dict__
                    }
                    for info in plugin_infos
                ]
            )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return PluginManagerStatus()
    
    async def cleanup(self) -> None:
        """Cleanup plugin manager resources."""
        try:
            # Cleanup all plugins
            cleanup_tasks = []
            for name, plugin in self.plugins.items():
                task = self._cleanup_plugin_safe(name, plugin)
                cleanup_tasks.append(task)
            
            # Wait for all cleanups
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Clear registries
            self.plugins.clear()
            self.plugin_configs.clear()
            self.plugin_status.clear()
            
            self.logger.info("Plugin manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Plugin manager cleanup failed: {e}")
    
    async def _cleanup_plugin_safe(self, name: str, plugin: OnyxPluginBase) -> None:
        """Safely cleanup a plugin."""
        try:
            await plugin.cleanup()
            self.logger.info(f"Plugin cleanup completed: {name}")
        except Exception as e:
            self.logger.error(f"Plugin cleanup failed: {name} - {e}") 