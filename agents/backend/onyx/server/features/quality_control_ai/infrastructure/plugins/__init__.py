"""
Plugin System

Extensible plugin architecture for quality control AI.
"""

from .base import BasePlugin, PluginRegistry
from .manager import PluginManager

__all__ = [
    "BasePlugin",
    "PluginRegistry",
    "PluginManager",
]



