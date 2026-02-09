"""
Plugin System Module

Provides:
- Plugin management
- Plugin loading
- Plugin registry
"""

from .plugin_manager import (
    PluginManager,
    load_plugin,
    register_plugin,
    get_plugin
)

from .plugin_base import (
    BasePlugin,
    PluginInterface
)

__all__ = [
    # Plugin management
    "PluginManager",
    "load_plugin",
    "register_plugin",
    "get_plugin",
    # Plugin base
    "BasePlugin",
    "PluginInterface"
]
