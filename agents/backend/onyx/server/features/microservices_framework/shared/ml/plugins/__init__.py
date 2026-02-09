"""
Plugins Module
Plugin system for extending functionality.
"""

from .plugin_manager import (
    Plugin,
    ModelPlugin,
    DataPlugin,
    TrainingPlugin,
    PluginManager,
)

__all__ = [
    "Plugin",
    "ModelPlugin",
    "DataPlugin",
    "TrainingPlugin",
    "PluginManager",
]



