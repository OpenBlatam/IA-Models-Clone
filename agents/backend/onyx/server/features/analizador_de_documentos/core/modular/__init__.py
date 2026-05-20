"""
Modular System - Sistema Modular
================================

Sistema modular para el Document Analyzer con mejor organización y separación de responsabilidades.
"""

from .module_manager import ModuleManager, ModuleRegistry
from .plugin_system import PluginSystem, PluginInterface
from .service_locator import ServiceLocator
from .event_bus import EventBus, Event

__all__ = [
    "ModuleManager",
    "ModuleRegistry",
    "PluginSystem",
    "PluginInterface",
    "ServiceLocator",
    "EventBus",
    "Event"
]


