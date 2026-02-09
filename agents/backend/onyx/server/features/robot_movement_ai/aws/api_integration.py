"""
Advanced API Integration (Modular)
==================================

Integrates all advanced features using modular plugin system.
"""

import logging
from fastapi import FastAPI
from robot_movement_ai.config.robot_config import RobotConfig

from aws.core.app_factory import create_modular_robot_app, AppFactory
from aws.core.config_manager import AppConfig

logger = logging.getLogger(__name__)


def create_advanced_robot_app(config: RobotConfig, app_config: AppConfig = None) -> FastAPI:
    """
    Create FastAPI app with all advanced features using modular plugin system.
    
    Args:
        config: Robot configuration
        app_config: Optional application configuration (uses env vars if not provided)
    
    Returns:
        Configured FastAPI app with all plugins
    """
    return create_modular_robot_app(config, app_config)


# Backward compatibility
def create_robot_app_with_plugins(config: RobotConfig) -> FastAPI:
    """Alias for create_advanced_robot_app."""
    return create_advanced_robot_app(config)

