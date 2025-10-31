"""
Plugin System
=============

This module provides a plugin system for extending functionality,
each component in its own focused module following the Single Responsibility Principle.
"""

from .plugin_manager import PluginManager
from .plugin_interface import PluginInterface
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader
from .plugin_validator import PluginValidator
from .plugin_config import PluginConfig

__all__ = [
    "PluginManager",
    "PluginInterface",
    "PluginRegistry",
    "PluginLoader",
    "PluginValidator",
    "PluginConfig"
]




