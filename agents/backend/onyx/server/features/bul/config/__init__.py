"""
BUL Configuration Module
========================

Configuration management for the BUL system.
"""

from .bul_config import (
    get_config,
    get_config_manager,
    ConfigManager,
    BULConfig,
    APIConfig,
    DatabaseConfig,
    ServerConfig,
    CacheConfig,
    LoggingConfig,
    Environment
)

def is_production() -> bool:
    """Check if running in production environment"""
    config = get_config()
    return config.environment == Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development environment"""
    config = get_config()
    return config.environment == Environment.DEVELOPMENT

def is_testing() -> bool:
    """Check if running in testing environment"""
    config = get_config()
    return config.environment == Environment.DEVELOPMENT  # Use development for testing

__all__ = [
    "get_config",
    "get_config_manager",
    "ConfigManager",
    "is_production",
    "is_development", 
    "is_testing",
    "BULConfig",
    "APIConfig",
    "DatabaseConfig",
    "ServerConfig",
    "CacheConfig",
    "LoggingConfig",
    "Environment"
]