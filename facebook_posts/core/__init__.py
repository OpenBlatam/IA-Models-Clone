#!/usr/bin/env python3
"""
Core Module - Ultra-Modular Architecture v3.7
Base classes and systems for all modules
"""

from .base_system import (
    BaseModule,
    ModuleConfig,
    ModuleState,
    ModulePriority,
    create_module_config,
    validate_module_dependencies
)

from .module_manager import ModuleManager
from .plugin_system import PluginManager, PluginInfo
from .config_manager import ConfigManager, ConfigFormat, ConfigSource, ConfigValidation
from .event_system import (
    EventSystem,
    Event,
    EventHandler,
    EventPriority,
    create_event_filter,
    create_priority_filter,
    create_source_filter
)

__version__ = "3.7.0"
__author__ = "Ultra-Modular AI System"
__description__ = "Core module system for ultra-modular architecture"

__all__ = [
    # Base system
    'BaseModule',
    'ModuleConfig', 
    'ModuleState',
    'ModulePriority',
    'create_module_config',
    'validate_module_dependencies',
    
    # Module management
    'ModuleManager',
    
    # Plugin system
    'PluginManager',
    'PluginInfo',
    
    # Configuration management
    'ConfigManager',
    'ConfigFormat',
    'ConfigSource',
    'ConfigValidation',
    
    # Event system
    'EventSystem',
    'Event',
    'EventHandler',
    'EventPriority',
    'create_event_filter',
    'create_priority_filter',
    'create_source_filter'
]

# Version info
VERSION_INFO = {
    'version': __version__,
    'major': 3,
    'minor': 7,
    'patch': 0,
    'release': 'stable'
}

# Module information
MODULE_INFO = {
    'name': 'core',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'components': [
        'BaseModule - Abstract base class for all modules',
        'ModuleManager - Central module management system',
        'PluginManager - Dynamic plugin loading and management',
        'ConfigManager - Hierarchical configuration management',
        'EventSystem - Pub/sub messaging and event handling'
    ]
}
