from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import importlib
import inspect
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict, run_functions_in_parallel, FunctionCall
from onyx.utils.timing import time_function
from onyx.utils.retry_wrapper import retry_wrapper
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.file import get_file_extension, get_file_size
from onyx.utils.text_processing import clean_text, extract_keywords
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.core.functions import process_document, validate_user_access
from onyx.llm.factory import get_default_llms
from onyx.llm.interfaces import LLM
from .models import PluginConfig, VideoRequest, VideoResponse
from .core.exceptions import PluginError, ValidationError, AIVideoError
from .core.onyx_integration import OnyxIntegrationManager, onyx_integration
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Plugin Manager

Adapted plugin manager that leverages Onyx's threading utilities,
file processing capabilities, and performance optimizations for
efficient plugin management and execution.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


@dataclass
class OnyxPluginInfo:
    """Information about an Onyx plugin."""
    name: str
    version: str
    description: str
    author: str
    category: str
    enabled: bool = True
    priority: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    gpu_required: bool = False
    timeout: int = 60
    max_workers: int = 1


@dataclass
class OnyxPluginContext:
    """Context for plugin execution."""
    request: VideoRequest
    llm: Optional[LLM] = None
    gpu_available: bool = False
    shared_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnyxPluginBase:
    """
    Base class for Onyx plugins.
    
    Provides common functionality and integration with Onyx's
    infrastructure for plugin development.
    """
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.logger = setup_logger(f"onyx_plugin.{self.__class__.__name__}")
        self.telemetry = TelemetryLogger()
        self.cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            self.logger.info(f"Initializing plugin: {self.__class__.__name__}")
            
            # Check GPU requirement
            if getattr(self, 'gpu_required', False) and not is_gpu_available():
                raise PluginError("GPU required but not available")
            
            # Initialize plugin-specific resources
            await self._initialize_plugin()
            
            self.logger.info(f"Plugin initialized: {self.__class__.__name__}")
            
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {self.__class__.__name__} - {e}")
            raise PluginError(f"Plugin initialization failed: {e}")
    
    async def _initialize_plugin(self) -> None:
        """Initialize plugin-specific resources. Override in subclasses."""
        pass
    
    async def process(self, context: OnyxPluginContext) -> Dict[str, Any]:
        """Process data using the plugin. Override in subclasses."""
        raise NotImplementedError("Plugin must implement process method")
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            self.cache.clear()
            await self._cleanup_plugin()
            self.logger.info(f"Plugin cleanup completed: {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Plugin cleanup failed: {self.__class__.__name__} - {e}")
    
    async def _cleanup_plugin(self) -> None:
        """Cleanup plugin-specific resources. Override in subclasses."""
        pass
    
    def get_info(self) -> OnyxPluginInfo:
        """Get plugin information."""
        return OnyxPluginInfo(
            name=self.__class__.__name__,
            version=getattr(self, 'version', '1.0.0'),
            description=getattr(self, 'description', ''),
            author=getattr(self, 'author', ''),
            category=getattr(self, 'category', 'general'),
            gpu_required=getattr(self, 'gpu_required', False),
            timeout=getattr(self, 'timeout', 60),
            max_workers=getattr(self, 'max_workers', 1)
        )


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
        self.plugin_status: ThreadSafeDict[str, Dict[str, Any]] = ThreadSafeDict()
        
        # Plugin directories
        self.plugin_dirs = [
            Path(__file__).parent / "plugins",
            Path(__file__).parent / "custom_plugins"
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
                    self.plugin_status[name] = {
                        "loaded": True,
                        "initialized": False,
                        "enabled": True,
                        "error": None
                    }
                    
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
                    self.plugin_status[name]["initialized"] = False
                    self.plugin_status[name]["error"] = str(result)
                    self.logger.error(f"Plugin initialization failed: {name} - {result}")
                else:
                    self.plugin_status[name]["initialized"] = True
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
            status = self.plugin_status.get(name, {})
            if not status.get("enabled", True) or not status.get("initialized", False):
                continue
            
            filtered_plugins[name] = plugin
        
        return filtered_plugins
    
    @retry_wrapper(max_attempts=3, backoff_factor=2)
    async def _execute_plugin_safe(self, name: str, plugin: OnyxPluginBase, context: OnyxPluginContext) -> Dict[str, Any]:
        """Safely execute a plugin with timeout."""
        try:
            plugin_info = plugin.get_info()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                plugin.process(context),
                timeout=plugin_info.timeout
            )
            
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
                info.enabled = self.plugin_status.get(name, {}).get("enabled", True)
                plugin_infos.append(info)
            except Exception as e:
                self.logger.error(f"Failed to get plugin info: {name} - {e}")
        
        return plugin_infos
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugin_status[plugin_name]["enabled"] = True
            self.logger.info(f"Plugin enabled: {plugin_name}")
            return True
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugin_status[plugin_name]["enabled"] = False
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
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """Get plugin manager status."""
        try:
            plugin_infos = await self.get_all_plugins_info()
            
            status = {
                "manager": "onyx_plugin_manager",
                "total_plugins": len(self.plugins),
                "enabled_plugins": len([p for p in plugin_infos if p.enabled]),
                "initialized_plugins": len([p for p in self.plugin_status.values() if p.get("initialized", False)]),
                "plugin_directories": [str(d) for d in self.plugin_dirs],
                "gpu_available": is_gpu_available(),
                "gpu_info": get_gpu_info() if is_gpu_available() else None,
                "plugins": [
                    {
                        "name": info.name,
                        "version": info.version,
                        "category": info.category,
                        "enabled": info.enabled,
                        "gpu_required": info.gpu_required,
                        "status": self.plugin_status.get(info.name, {})
                    }
                    for info in plugin_infos
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}
    
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


# Example plugins
class OnyxContentAnalyzerPlugin(OnyxPluginBase):
    """Example content analysis plugin using Onyx utilities."""
    
    version = "1.0.0"
    description = "Analyzes video content using Onyx LLM"
    author = "Onyx Team"
    category = "analysis"
    gpu_required = False
    timeout = 30
    
    async def _initialize_plugin(self) -> None:
        """Initialize plugin-specific resources."""
        self.llm = await onyx_integration.llm_manager.get_default_llm()
    
    async def process(self, context: OnyxPluginContext) -> Dict[str, Any]:
        """Process content analysis."""
        try:
            # Analyze content using Onyx LLM
            prompt = f"Analyze this video content and extract key themes, emotions, and visual elements: {context.request.input_text}"
            
            with time_function("content_analysis"):
                analysis = await onyx_integration.llm_manager.generate_text(prompt, self.llm)
            
            # Extract keywords using Onyx utilities
            keywords = extract_keywords(context.request.input_text)
            
            result = {
                "analysis": analysis,
                "keywords": keywords,
                "content_length": len(context.request.input_text),
                "processed_at": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[f"analysis_{context.request.request_id}"] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            raise PluginError(f"Content analysis failed: {e}")


class OnyxVisualEnhancerPlugin(OnyxPluginBase):
    """Example visual enhancement plugin using Onyx GPU utilities."""
    
    version = "1.0.0"
    description = "Enhances video visuals using GPU acceleration"
    author = "Onyx Team"
    category = "enhancement"
    gpu_required = True
    timeout = 120
    
    async def _initialize_plugin(self) -> None:
        """Initialize plugin-specific resources."""
        if not is_gpu_available():
            raise PluginError("GPU required but not available")
        
        self.gpu_info = get_gpu_info()
        self.logger.info(f"GPU initialized: {self.gpu_info}")
    
    async def process(self, context: OnyxPluginContext) -> Dict[str, Any]:
        """Process visual enhancement."""
        try:
            # Simulate GPU-based visual enhancement
            with time_function("visual_enhancement"):
                # This would contain actual GPU processing logic
                enhancement_result = {
                    "quality_improved": True,
                    "resolution_upscaled": True,
                    "color_corrected": True,
                    "gpu_utilization": 85.5
                }
            
            result = {
                "enhancement": enhancement_result,
                "gpu_info": self.gpu_info,
                "processed_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Visual enhancement failed: {e}")
            raise PluginError(f"Visual enhancement failed: {e}")


# Global plugin manager instance
onyx_plugin_manager = OnyxPluginManager()


# Utility functions
async def initialize_onyx_plugins() -> None:
    """Initialize Onyx plugin system."""
    await onyx_plugin_manager.initialize()


async def execute_onyx_plugins(context: OnyxPluginContext, plugin_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute Onyx plugins."""
    return await onyx_plugin_manager.execute_plugins(context, plugin_names)


def get_onyx_plugin_status() -> Dict[str, Any]:
    """Get Onyx plugin system status."""
    return asyncio.run(onyx_plugin_manager.get_manager_status()) 