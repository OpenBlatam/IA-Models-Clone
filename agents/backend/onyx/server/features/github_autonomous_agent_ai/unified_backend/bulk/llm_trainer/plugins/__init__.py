"""
Plugins Module
==============

Plugin system for extending trainer functionality.
Allows for custom components and extensions.

Author: BUL System
Date: 2024
"""

from .base_plugin import BasePlugin, PluginRegistry
from .callback_plugin import CallbackPlugin
from .metric_plugin import MetricPlugin

__all__ = [
    "BasePlugin",
    "PluginRegistry",
    "CallbackPlugin",
    "MetricPlugin",
]

