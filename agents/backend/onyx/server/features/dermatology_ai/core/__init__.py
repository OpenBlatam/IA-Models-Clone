"""
Core module for modular architecture
Provides plugin system, module loading, and service factory
"""

from .plugin_system import (
    BasePlugin,
    PluginRegistry,
    PluginType,
    PluginMetadata,
    get_plugin_registry,
)

from .module_loader import (
    ModuleLoader,
    get_module_loader,
)

from .service_factory import (
    ServiceFactory,
    ServiceScope,
    ServiceDescriptor,
    get_service_factory,
)

__all__ = [
    # Plugin System
    "BasePlugin",
    "PluginRegistry",
    "PluginType",
    "PluginMetadata",
    "get_plugin_registry",
    # Module Loader
    "ModuleLoader",
    "get_module_loader",
    # Service Factory
    "ServiceFactory",
    "ServiceScope",
    "ServiceDescriptor",
    "get_service_factory",
]
