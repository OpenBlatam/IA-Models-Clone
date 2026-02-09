from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
    from ai_video.plugins import PluginManager, ManagerConfig
from .base import BasePlugin, PluginMetadata
from .loader import PluginLoader, LoadResult
from .validator import PluginValidator, ValidationLevel, ValidationStatus, ValidationReport
from .registry import PluginRegistry, PluginState, RegistryEntry
from .manager import PluginManager, ManagerConfig
        import asyncio
        import asyncio
   from ai_video.plugins import PluginManager, ManagerConfig
   from ai_video.plugins import quick_start
   from ai_video.plugins import get_plugin_info, list_available_plugins
   from ai_video.plugins import PluginManager, ValidationLevel
from ai_video.plugins import BasePlugin, PluginMetadata
from typing import Any, List, Dict, Optional
import logging
"""
AI Video Plugin System

A comprehensive, production-ready plugin system for AI video generation.

This module provides:
- Plugin base classes and interfaces
- Plugin loading and validation
- Plugin registry and management
- Event handling and lifecycle management
- Configuration management
- Performance monitoring

Quick Start:
    
    # Create and start the plugin manager
    config = ManagerConfig(auto_discover=True, auto_load=True)
    manager = PluginManager(config)
    await manager.start()
    
    # Load a specific plugin
    plugin = await manager.load_plugin("my_plugin", {"key": "value"})
    
    # Get plugin information
    info = manager.get_plugin_info("my_plugin")
    print(f"Plugin: {info.name} v{info.version}")
    
    # List all plugins
    plugins = manager.list_plugins()
    print(f"Loaded plugins: {plugins}")
    
    # Get statistics
    stats = manager.get_stats()
    print(f"Plugin stats: {stats}")
"""


__all__ = [
    # Core classes
    'BasePlugin',
    'PluginMetadata',
    
    # Loading and validation
    'PluginLoader',
    'LoadResult',
    'PluginValidator',
    'ValidationLevel',
    'ValidationStatus',
    'ValidationReport',
    
    # Registry and management
    'PluginRegistry',
    'PluginState',
    'RegistryEntry',
    'PluginManager',
    'ManagerConfig',
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Video Team"
__description__ = "Production-ready plugin system for AI video generation"

# Convenience functions for common operations

async def create_plugin_manager(
    auto_discover: bool = True,
    auto_load: bool = False,
    auto_initialize: bool = False,
    validation_level: str = "standard",
    plugin_dirs: list = None,
    **kwargs
) -> PluginManager:
    """
    Create and configure a plugin manager with common settings.
    
    Args:
        auto_discover: Whether to automatically discover plugins
        auto_load: Whether to automatically load discovered plugins
        auto_initialize: Whether to automatically initialize loaded plugins
        validation_level: Validation level ("basic", "standard", "strict", "security")
        plugin_dirs: Directories to search for plugins
        **kwargs: Additional configuration options
        
    Returns:
        Configured PluginManager instance
    """
    # Convert validation level string to enum
    level_map = {
        "basic": ValidationLevel.BASIC,
        "standard": ValidationLevel.STANDARD,
        "strict": ValidationLevel.STRICT,
        "security": ValidationLevel.SECURITY
    }
    val_level = level_map.get(validation_level.lower(), ValidationLevel.STANDARD)
    
    config = ManagerConfig(
        auto_discover=auto_discover,
        auto_load=auto_load,
        auto_initialize=auto_initialize,
        validation_level=val_level,
        plugin_dirs=plugin_dirs,
        **kwargs
    )
    
    return PluginManager(config)


async def quick_start() -> PluginManager:
    """
    Quick start function that creates a plugin manager with recommended settings.
    
    Returns:
        Started PluginManager instance
    """
    manager = await create_plugin_manager(
        auto_discover=True,
        auto_load=True,
        auto_initialize=True,
        validation_level="standard"
    )
    
    success = await manager.start()
    if not success:
        raise RuntimeError("Failed to start plugin manager")
    
    return manager


def get_plugin_info(plugin_name: str, manager: PluginManager = None) -> dict:
    """
    Get information about a plugin.
    
    Args:
        plugin_name: Name of the plugin
        manager: PluginManager instance (creates one if not provided)
        
    Returns:
        Plugin information dictionary
    """
    if manager is None:
        # Create a temporary manager for discovery
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create manager
            manager = loop.create_task(create_plugin_manager(auto_load=False))
            manager = loop.run_until_complete(manager)
            loop.run_until_complete(manager.discover_plugins())
        else:
            # We're in a sync context, use asyncio.run
            async def _get_info():
                
    """_get_info function."""
m = await create_plugin_manager(auto_load=False)
                await m.discover_plugins()
                return m.get_plugin_info(plugin_name)
            
            info = asyncio.run(_get_info())
            return info.to_dict() if info else None
    
    info = manager.get_plugin_info(plugin_name)
    return info.to_dict() if info else None


def list_available_plugins(manager: PluginManager = None) -> list:
    """
    List all available plugins.
    
    Args:
        manager: PluginManager instance (creates one if not provided)
        
    Returns:
        List of plugin names
    """
    if manager is None:
        # Create a temporary manager for discovery
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create manager
            manager = loop.create_task(create_plugin_manager(auto_load=False))
            manager = loop.run_until_complete(manager)
            loop.run_until_complete(manager.discover_plugins())
        else:
            # We're in a sync context, use asyncio.run
            async def _list_plugins():
                
    """_list_plugins function."""
m = await create_plugin_manager(auto_load=False)
                await m.discover_plugins()
                return list(m.registry.discovered_plugins.keys())
            
            return asyncio.run(_list_plugins())
    
    return list(manager.registry.discovered_plugins.keys())


# Example usage and documentation

__doc__ = """
AI Video Plugin System
======================

A comprehensive, production-ready plugin system for AI video generation.

Features:
---------
- 🔌 Plugin discovery and loading
- ✅ Validation and error handling
- 🔄 Lifecycle management
- 📊 Performance monitoring
- 🎯 Event handling
- ⚙️ Configuration management
- 🔒 Security validation

Quick Examples:
--------------

1. Basic Usage:
   ```python
   
   # Create manager
   config = ManagerConfig(auto_discover=True, auto_load=True)
   manager = PluginManager(config)
   await manager.start()
   
   # Load a plugin
   plugin = await manager.load_plugin("my_plugin", {"key": "value"})
   ```

2. Quick Start:
   ```python
   
   # Start with recommended settings
   manager = await quick_start()
   
   # List loaded plugins
   plugins = manager.list_plugins()
   print(f"Loaded: {plugins}")
   ```

3. Plugin Information:
   ```python
   
   # List all available plugins
   available = list_available_plugins()
   print(f"Available: {available}")
   
   # Get plugin info
   info = get_plugin_info("my_plugin")
   print(f"Info: {info}")
   ```

4. Advanced Usage:
   ```python
   
   # Create manager with strict validation
   config = ManagerConfig(
       auto_discover=True,
       auto_load=True,
       validation_level=ValidationLevel.STRICT,
       enable_events=True,
       enable_metrics=True
   )
   
   manager = PluginManager(config)
   await manager.start()
   
   # Add event handlers
   def on_plugin_loaded(plugin_name, plugin) -> Any:
       print(f"Plugin loaded: {plugin_name}")
   
   manager.add_event_handler("plugin_loaded", on_plugin_loaded)
   
   # Get health report
   health = manager.get_health_report()
   print(f"Health: {health}")
   ```

Plugin Development:
------------------

To create a plugin, inherit from BasePlugin:

```python

class MyPlugin(BasePlugin):
    def __init__(self, config=None) -> Any:
        super().__init__(config)
        self.name = "my_plugin"
        self.version = "1.0.0"
        self.description = "My awesome plugin"
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author="Your Name",
            category="extractor"
        )
    
    async def initialize(self) -> Any:
        # Initialize your plugin
        pass
    
    async def cleanup(self) -> Any:
        # Cleanup your plugin
        pass
```

For more information, see the individual module documentation.
""" 