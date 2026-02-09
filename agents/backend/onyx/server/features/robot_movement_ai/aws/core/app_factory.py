"""
Application Factory
==================

Creates FastAPI application with modular plugin system.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from robot_movement_ai.config.robot_config import RobotConfig

from aws.core.config_manager import AppConfig
from aws.core.plugin_manager import PluginManager
from aws.plugins.middleware import (
    TracingMiddlewarePlugin,
    RateLimitingMiddlewarePlugin,
    CircuitBreakerMiddlewarePlugin,
    CachingMiddlewarePlugin,
    LoggingMiddlewarePlugin,
    SecurityHeadersMiddlewarePlugin,
)
from aws.plugins.monitoring import PrometheusMonitoringPlugin
from aws.plugins.security import OAuth2SecurityPlugin
from aws.plugins.messaging import KafkaMessagingPlugin

logger = logging.getLogger(__name__)


class AppFactory:
    """Factory for creating FastAPI applications with plugins."""
    
    def __init__(self, app_config: Optional[AppConfig] = None):
        """Initialize factory."""
        self.app_config = app_config or AppConfig.from_env()
        self.plugin_manager = PluginManager(self.app_config.to_dict())
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        """Register default plugins."""
        # Middleware plugins
        self.plugin_manager.register_plugin(TracingMiddlewarePlugin())
        self.plugin_manager.register_plugin(RateLimitingMiddlewarePlugin())
        self.plugin_manager.register_plugin(CircuitBreakerMiddlewarePlugin())
        self.plugin_manager.register_plugin(CachingMiddlewarePlugin())
        self.plugin_manager.register_plugin(LoggingMiddlewarePlugin())
        self.plugin_manager.register_plugin(SecurityHeadersMiddlewarePlugin())
        
        # Monitoring plugins
        self.plugin_manager.register_plugin(PrometheusMonitoringPlugin())
        
        # Security plugins
        self.plugin_manager.register_plugin(OAuth2SecurityPlugin())
        
        # Messaging plugins
        if self.app_config.messaging.enable_kafka:
            self.plugin_manager.register_plugin(KafkaMessagingPlugin())
    
    def create_app(self, robot_config: RobotConfig) -> FastAPI:
        """Create FastAPI application with all plugins."""
        from robot_movement_ai.api.robot_api import create_robot_app
        
        # Create base app
        app = create_robot_app(robot_config)
        
        # Setup all plugins
        app = self.plugin_manager.setup_plugins(app)
        
        # Store plugin manager in app state
        app.state.plugin_manager = self.plugin_manager
        app.state.app_config = self.app_config
        
        logger.info("Application created with modular plugin system")
        
        return app
    
    def get_plugin_manager(self) -> PluginManager:
        """Get plugin manager."""
        return self.plugin_manager
    
    def get_config(self) -> AppConfig:
        """Get application configuration."""
        return self.app_config


def create_modular_robot_app(robot_config: RobotConfig, app_config: Optional[AppConfig] = None) -> FastAPI:
    """
    Create modular FastAPI application.
    
    Args:
        robot_config: Robot configuration
        app_config: Optional application configuration (uses env vars if not provided)
    
    Returns:
        Configured FastAPI app
    """
    factory = AppFactory(app_config)
    return factory.create_app(robot_config)















