"""
Plugin System for Extensibility
Allows adding custom functionality without modifying core code
"""

from .base import BasePlugin, PluginRegistry
from .manager import PluginManager

__all__ = [
    "BasePlugin",
    "PluginRegistry",
    "PluginManager",
]
