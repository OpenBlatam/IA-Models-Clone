"""
Core Module
===========

Core interfaces, plugin management, and application factory.
"""

from aws.core.interfaces import (
    MiddlewarePlugin,
    MonitoringPlugin,
    SecurityPlugin,
    MessagingPlugin,
    CachePlugin,
    WorkerPlugin,
    ServiceRegistry,
)
from aws.core.plugin_manager import PluginManager, PluginRegistry
from aws.core.config_manager import (
    AppConfig,
    MiddlewareConfig,
    MonitoringConfig,
    SecurityConfig,
    MessagingConfig,
    WorkerConfig,
    CacheConfig,
)
from aws.core.app_factory import AppFactory, create_modular_robot_app

__all__ = [
    "MiddlewarePlugin",
    "MonitoringPlugin",
    "SecurityPlugin",
    "MessagingPlugin",
    "CachePlugin",
    "WorkerPlugin",
    "ServiceRegistry",
    "PluginManager",
    "PluginRegistry",
    "AppConfig",
    "MiddlewareConfig",
    "MonitoringConfig",
    "SecurityConfig",
    "MessagingConfig",
    "WorkerConfig",
    "CacheConfig",
    "AppFactory",
    "create_modular_robot_app",
]















