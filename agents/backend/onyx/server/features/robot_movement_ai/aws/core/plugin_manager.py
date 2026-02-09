"""
Plugin Manager
==============

Manages all plugins using a registry pattern.
"""

import logging
from typing import Dict, List, Type, Optional, Any
from aws.core.interfaces import (
    MiddlewarePlugin,
    MonitoringPlugin,
    SecurityPlugin,
    MessagingPlugin,
    CachePlugin,
    WorkerPlugin,
    ServiceRegistry
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for all plugins."""
    
    def __init__(self):
        self._middleware_plugins: Dict[str, MiddlewarePlugin] = {}
        self._monitoring_plugins: Dict[str, MonitoringPlugin] = {}
        self._security_plugins: Dict[str, SecurityPlugin] = {}
        self._messaging_plugins: Dict[str, MessagingPlugin] = {}
        self._cache_plugins: Dict[str, CachePlugin] = {}
        self._worker_plugins: Dict[str, WorkerPlugin] = {}
        self._services: Dict[str, Any] = {}
    
    def register_middleware(self, plugin: MiddlewarePlugin) -> None:
        """Register middleware plugin."""
        name = plugin.get_name()
        self._middleware_plugins[name] = plugin
        logger.info(f"Registered middleware plugin: {name}")
    
    def register_monitoring(self, plugin: MonitoringPlugin) -> None:
        """Register monitoring plugin."""
        name = plugin.get_name()
        self._monitoring_plugins[name] = plugin
        logger.info(f"Registered monitoring plugin: {name}")
    
    def register_security(self, plugin: SecurityPlugin) -> None:
        """Register security plugin."""
        name = plugin.get_name()
        self._security_plugins[name] = plugin
        logger.info(f"Registered security plugin: {name}")
    
    def register_messaging(self, plugin: MessagingPlugin) -> None:
        """Register messaging plugin."""
        name = plugin.get_name()
        self._messaging_plugins[name] = plugin
        logger.info(f"Registered messaging plugin: {name}")
    
    def register_cache(self, plugin: CachePlugin) -> None:
        """Register cache plugin."""
        name = plugin.get_name()
        self._cache_plugins[name] = plugin
        logger.info(f"Registered cache plugin: {name}")
    
    def register_worker(self, plugin: WorkerPlugin) -> None:
        """Register worker plugin."""
        name = plugin.get_name()
        self._worker_plugins[name] = plugin
        logger.info(f"Registered worker plugin: {name}")
    
    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self._services[name] = service
        logger.info(f"Registered service: {name}")
    
    def get_middleware_plugins(self) -> Dict[str, MiddlewarePlugin]:
        """Get all middleware plugins."""
        return self._middleware_plugins.copy()
    
    def get_monitoring_plugins(self) -> Dict[str, MonitoringPlugin]:
        """Get all monitoring plugins."""
        return self._monitoring_plugins.copy()
    
    def get_security_plugins(self) -> Dict[str, SecurityPlugin]:
        """Get all security plugins."""
        return self._security_plugins.copy()
    
    def get_messaging_plugin(self, name: Optional[str] = None) -> Optional[MessagingPlugin]:
        """Get messaging plugin (default or by name)."""
        if name:
            return self._messaging_plugins.get(name)
        return next(iter(self._messaging_plugins.values()), None)
    
    def get_cache_plugin(self, name: Optional[str] = None) -> Optional[CachePlugin]:
        """Get cache plugin (default or by name)."""
        if name:
            return self._cache_plugins.get(name)
        return next(iter(self._cache_plugins.values()), None)
    
    def get_worker_plugin(self, name: Optional[str] = None) -> Optional[WorkerPlugin]:
        """Get worker plugin (default or by name)."""
        if name:
            return self._worker_plugins.get(name)
        return next(iter(self._worker_plugins.values()), None)
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get service by name."""
        return self._services.get(name)
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self._services.keys())


class PluginManager:
    """Manages plugin lifecycle and configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = PluginRegistry()
        self._initialized = False
    
    def register_plugin(self, plugin: Any) -> None:
        """Register a plugin (auto-detects type)."""
        if isinstance(plugin, MiddlewarePlugin):
            self.registry.register_middleware(plugin)
        elif isinstance(plugin, MonitoringPlugin):
            self.registry.register_monitoring(plugin)
        elif isinstance(plugin, SecurityPlugin):
            self.registry.register_security(plugin)
        elif isinstance(plugin, MessagingPlugin):
            self.registry.register_messaging(plugin)
        elif isinstance(plugin, CachePlugin):
            self.registry.register_cache(plugin)
        elif isinstance(plugin, WorkerPlugin):
            self.registry.register_worker(plugin)
        else:
            logger.warning(f"Unknown plugin type: {type(plugin)}")
    
    def setup_plugins(self, app, plugin_type: str = "all") -> None:
        """Setup all enabled plugins."""
        if plugin_type == "all" or plugin_type == "middleware":
            for name, plugin in self.registry.get_middleware_plugins().items():
                if plugin.is_enabled(self.config):
                    try:
                        app = plugin.setup(app, self.config)
                        logger.info(f"Setup middleware plugin: {name}")
                    except Exception as e:
                        logger.error(f"Failed to setup middleware plugin {name}: {e}")
        
        if plugin_type == "all" or plugin_type == "monitoring":
            for name, plugin in self.registry.get_monitoring_plugins().items():
                if plugin.is_enabled(self.config):
                    try:
                        app = plugin.setup(app, self.config)
                        logger.info(f"Setup monitoring plugin: {name}")
                    except Exception as e:
                        logger.error(f"Failed to setup monitoring plugin {name}: {e}")
        
        if plugin_type == "all" or plugin_type == "security":
            for name, plugin in self.registry.get_security_plugins().items():
                if plugin.is_enabled(self.config):
                    try:
                        app = plugin.setup(app, self.config)
                        logger.info(f"Setup security plugin: {name}")
                    except Exception as e:
                        logger.error(f"Failed to setup security plugin {name}: {e}")
        
        self._initialized = True
        return app
    
    def get_registry(self) -> PluginRegistry:
        """Get plugin registry."""
        return self.registry















